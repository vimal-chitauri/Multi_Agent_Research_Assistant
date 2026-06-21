import hashlib
import os
import re
import time
import requests
import urllib.parse
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from dotenv import load_dotenv
from agents.base_agent import BaseAgent, AgentResult
from huggingface_hub import InferenceClient

load_dotenv(override=True)

IMAGES_DIR = Path(__file__).parent.parent / "generated_images"
IMAGES_DIR.mkdir(exist_ok=True)

IMAGE_W = 1080
IMAGE_H = 1920

# How many visual variants to generate per slide (different seeds → more choice)
NUM_VARIANTS = 2

HF_MODELS = [
    "black-forest-labs/FLUX.1-schnell",           # fastest free, great quality
    "stabilityai/stable-diffusion-3.5-medium",    # best anatomy on free tier
    "stabilityai/stable-diffusion-xl-base-1.0",
    "stabilityai/stable-diffusion-2-1",
    "runwayml/stable-diffusion-v1-5",
    "dreamlike-art/dreamlike-photoreal-2.0",
    "prompthero/openjourney",
]

POLLINATIONS_MODELS = [
    "flux-realism",   # most relevant / photorealistic — first choice
    "flux",
    "flux-anime",
    "flux-3d",
    "turbo",
    "dreamshaper",
]

# Anatomy-focused negative prompt — suppresses distorted hands/faces even when
# they slip through the "no people" instruction in the positive prompt.
NEGATIVE_PROMPT = (
    "deformed, bad anatomy, disfigured, poorly drawn face, mutation, mutated, "
    "extra limb, ugly, poorly drawn hands, missing limb, floating limbs, "
    "disconnected limbs, malformed hands, blurry, mutated hands and fingers, "
    "watermark, oversaturated, distorted hands, amputation, missing hands, "
    "extra fingers, fused fingers, too many fingers, long neck, bad proportions, "
    "duplicate, morbid, gross proportions, cloned face, out of frame, "
    "low quality, low resolution, text, logo, signature, username"
)


class ImageAgent(BaseAgent):

    def __init__(self):
        super().__init__(name="ImageAgent", model="fast")
        self.hf_token = os.getenv("HUGGINGFACE_TOKEN")

    @property
    def system_prompt(self) -> str:
        return "You are an image prompt engineer for Instagram Reels short-form video content."

    def run(self, input_data: dict) -> AgentResult:
        posts = input_data.get("posts", [])
        if not posts:
            return self._failure("No posts received from ContentAgent.")

        if self.hf_token:
            self._log(
                f"Image chain: [bold]{len(HF_MODELS)} HF[/bold] → "
                f"[bold]{len(POLLINATIONS_MODELS)} Pollinations[/bold] → "
                f"[bold]Openverse[/bold] → gradient  |  "
                f"9:16 ({IMAGE_W}×{IMAGE_H})  |  {NUM_VARIANTS} variants/slide"
            )
        else:
            self._log(
                f"[yellow]No HF token[/yellow] — chain: Pollinations → Openverse → gradient  |  "
                f"9:16 ({IMAGE_W}×{IMAGE_H})  |  {NUM_VARIANTS} variants/slide"
            )

        images_generated = 0

        for post in posts:
            topic = post["topic"]
            self._log(f"Generating images for: [cyan]{topic}[/cyan]")

            if "image_prompts" in post and isinstance(post["image_prompts"], list):
                raw_prompts = post["image_prompts"][:3]
                all_variants = self._generate_slides_parallel(raw_prompts, post["rank"], topic)

                # Primary path per slide (first variant) — used by video agent
                primary_paths = [v[0] if v else None for v in all_variants]
                # All flat paths across slides/variants — for UI display
                all_paths = [p for variants in all_variants for p in variants]

                post["image_paths"]    = [p for p in primary_paths if p]
                post["image_variants"] = all_variants
                post["image_path"]     = post["image_paths"][0] if post["image_paths"] else None
                post["image_status"]   = "generated" if post["image_paths"] else "failed"
                images_generated      += len(all_paths)
                if not post["image_paths"]:
                    self._log(f"  [red]All slides failed for: {topic}[/red]")
                else:
                    self._log(
                        f"  [green]{len(all_paths)} image(s) across "
                        f"{len(post['image_paths'])} slide(s) "
                        f"({NUM_VARIANTS} variants each)[/green]"
                    )
            else:
                raw      = post.get("image_prompt", topic)
                prompt   = self._rewrite_prompt(raw, topic)
                enhanced = self._enhance_prompt(prompt)
                cached   = self._cache_path(enhanced)
                if cached.exists():
                    self._log(f"  Cache hit → {cached.name}")
                    path = cached
                else:
                    filename = f"post_{post['rank']}_{int(time.time() * 1000)}.png"
                    path = self._generate(enhanced, filename, search_query=topic)
                if path:
                    post["image_path"]     = str(path)
                    post["image_paths"]    = [str(path)]
                    post["image_variants"] = [[str(path)]]
                    post["image_status"]   = "generated"
                    images_generated      += 1
                else:
                    post["image_path"]     = None
                    post["image_paths"]    = []
                    post["image_variants"] = []
                    post["image_status"]   = "failed"
                    self._log(f"  [red]All backends failed for: {topic}[/red]")

        self._log(f"[green]{images_generated} total image(s) across {len(posts)} post(s)[/green]")
        return self._success(
            data={"posts": posts, "total": len(posts), "images_generated": images_generated},
            reasoning=f"{images_generated} images via HF + Pollinations chain ({NUM_VARIANTS} variants/slide)"
        )

    def _generate_slides_parallel(
        self, raw_prompts: list[str], rank: int, topic: str
    ) -> list[list[str]]:
        """Generate NUM_VARIANTS per slide, all slides in parallel. Returns list[list[path]]."""
        # results[slide_idx] = list of successful paths
        results: list[list[str]] = [[] for _ in raw_prompts]

        def _do(slide_idx: int, variant_idx: int, raw: str) -> tuple[int, str | None]:
            prompt   = self._rewrite_prompt(raw, topic)
            enhanced = self._enhance_prompt(prompt)
            cache_key = f"{enhanced}|v{variant_idx}"
            cached    = self._cache_path(cache_key)
            if cached.exists():
                self._log(f"  Slide {slide_idx + 1} v{variant_idx + 1}: cache hit → {cached.name}")
                return slide_idx, str(cached)
            # Each variant uses a distinct seed so images look different
            seed     = variant_idx * 31337
            ts       = int(time.time() * 1000) + variant_idx
            filename = f"post_{rank}_{ts}_{slide_idx}_v{variant_idx}.png"
            path = self._generate(
                enhanced, filename,
                search_query=topic, slide_index=slide_idx, seed=seed,
            )
            if not path:
                self._log(f"  [red]Slide {slide_idx + 1} v{variant_idx + 1}: all backends failed[/red]")
            return slide_idx, str(path) if path else None

        tasks = [(i, v) for i in range(len(raw_prompts)) for v in range(NUM_VARIANTS)]
        with ThreadPoolExecutor(max_workers=min(len(tasks), 6)) as pool:
            futures = {pool.submit(_do, i, v, raw_prompts[i]): (i, v) for i, v in tasks}
            for fut in as_completed(futures):
                try:
                    slide_idx, path = fut.result()
                    if path:
                        results[slide_idx].append(path)
                except Exception as e:
                    self._log(f"  [red]Slide error: {e}[/red]")

        return results

    def _cache_path(self, key: str) -> Path:
        h = hashlib.md5(key.encode()).hexdigest()[:12]
        return IMAGES_DIR / f"cache_{h}.png"

    def _rewrite_prompt(self, raw_prompt: str, topic: str) -> str:
        llm_prompt = f"""You are an image prompt engineer for Instagram Reels (9:16 vertical).
Rewrite the image idea below into a vivid, specific scene for an AI image generator.

Topic: {topic}
Raw idea: {raw_prompt}

Rules:
- Describe ONE concrete visual scene: setting, lighting, key objects, mood, color palette
- AVOID all human figures, faces, hands, body parts — use objects, landscapes, or
  symbolic/abstract imagery instead (glowing tech, cityscapes, nature, icons, data visuals)
- No text, logos, watermarks in the scene
- Optimized for vertical 9:16 portrait framing
- 1-2 sentences, pure visual description only

Respond with ONLY the rewritten prompt, nothing else."""
        try:
            result = self.think(llm_prompt).strip()
            result = re.sub(r'^["\']|["\']$', '', result).strip()
            return result if result else raw_prompt
        except Exception:
            return raw_prompt

    _BLOCKED_TERMS = [
        "President", "Prime Minister", "Governor", "Senator", "Secretary of State",
        "Trump", "Biden", "Obama", "Putin", "Xi Jinping", "Netanyahu",
        "Zelensky", "Macron", "Modi", "Kim Jong", "Erdogan", "Khamenei",
        "LeBron", "NBA star", "NFL player", "celebrity", "politician",
        "Iranian leader", "Iranian official", "world leader", "the president",
    ]

    def _enhance_prompt(self, raw_prompt: str) -> str:
        safe = raw_prompt
        for term in self._BLOCKED_TERMS:
            safe = re.sub(r'\b' + re.escape(term) + r'\b', "a public figure",
                          safe, flags=re.IGNORECASE)
        return (
            f"{safe}, "
            "no real people, no faces, no hands, symbolic imagery, "
            "professional photography, vibrant colors, high resolution, "
            "9:16 vertical portrait, sharp focus, cinematic lighting, modern aesthetic"
        )

    def _generate(self, prompt: str, filename: str,
                  model_index: int = 0, _rate_retries: int = 0,
                  search_query: str = None, slide_index: int = 0,
                  seed: int = 0) -> Path | None:
        if not self.hf_token or model_index >= len(HF_MODELS):
            return self._generate_pollinations(
                prompt, filename, 0,
                search_query=search_query, slide_index=slide_index, seed=seed,
            )

        model = HF_MODELS[model_index]
        try:
            client = InferenceClient(token=self.hf_token)
            self._log(f"  HF [{model_index + 1}/{len(HF_MODELS)}]: {model.split('/')[-1]}")
            image = client.text_to_image(
                prompt,
                model=model,
                width=IMAGE_W,
                height=IMAGE_H,
                negative_prompt=NEGATIVE_PROMPT,
                seed=seed or None,
            )
            image_path = IMAGES_DIR / filename
            image.save(str(image_path))
            self._log(f"  [green]HF OK → {filename}[/green]")
            return image_path

        except Exception as e:
            err = str(e)
            self._log(f"  [yellow]{model.split('/')[-1]}: {err[:140]}[/yellow]")

            if "503" in err or "loading" in err.lower():
                self._log("  Model loading — waiting 25s...")
                time.sleep(25)
                return self._generate(prompt, filename, model_index, _rate_retries,
                                      search_query, slide_index, seed)

            elif "429" in err or "rate" in err.lower() or "too many" in err.lower():
                if _rate_retries < 3:
                    wait = 30 * (_rate_retries + 1)
                    self._log(f"  Rate limited — backing off {wait}s ({_rate_retries + 1}/3)...")
                    time.sleep(wait)
                    return self._generate(prompt, filename, model_index, _rate_retries + 1,
                                          search_query, slide_index, seed)
                self._log("  Rate limit exhausted — next HF model...")
                return self._generate(prompt, filename, model_index + 1, 0,
                                      search_query, slide_index, seed)

            elif "402" in err or "payment" in err.lower():
                self._log("  Pro plan required — next HF model...")
                return self._generate(prompt, filename, model_index + 1, 0,
                                      search_query, slide_index, seed)

            elif ("nsfw" in err.lower() or "safety" in err.lower()
                  or "inappropriate" in err.lower() or "400" in err):
                self._log("  Safety filter — next HF model...")
                return self._generate(prompt, filename, model_index + 1, 0,
                                      search_query, slide_index, seed)

            else:
                self._log("  Error — next HF model...")
                return self._generate(prompt, filename, model_index + 1, 0,
                                      search_query, slide_index, seed)

    def _generate_pollinations(self, prompt: str, filename: str,
                                model_index: int = 0,
                                search_query: str = None,
                                slide_index: int = 0,
                                seed: int = 0) -> Path | None:
        if model_index >= len(POLLINATIONS_MODELS):
            self._log("  Pollinations exhausted — trying Openverse...")
            return self._generate_openverse(filename, search_query=search_query,
                                            slide_index=slide_index)

        model = POLLINATIONS_MODELS[model_index]
        self._log(f"  Pollinations [{model_index + 1}/{len(POLLINATIONS_MODELS)}]: {model}")

        try:
            encoded          = urllib.parse.quote(prompt[:480])
            encoded_negative = urllib.parse.quote(NEGATIVE_PROMPT[:300])
            seed_val         = (seed or int(time.time())) + model_index
            url = (
                f"https://image.pollinations.ai/prompt/{encoded}"
                f"?width={IMAGE_W}&height={IMAGE_H}&model={model}"
                f"&negative={encoded_negative}&seed={seed_val}&enhance=true&nologo=true"
            )
            r = requests.get(url, timeout=90, stream=True)

            if r.status_code == 402:
                self._log(f"  Rate-limited ({model}) — next variant...")
                return self._generate_pollinations(prompt, filename, model_index + 1,
                                                   search_query, slide_index, seed)
            if r.status_code != 200:
                self._log(f"  HTTP {r.status_code} — next variant...")
                return self._generate_pollinations(prompt, filename, model_index + 1,
                                                   search_query, slide_index, seed)

            image_path = IMAGES_DIR / filename
            with open(image_path, "wb") as f:
                for chunk in r.iter_content(8192):
                    f.write(chunk)

            from PIL import Image as PILImage
            try:
                with PILImage.open(str(image_path)) as img:
                    img.verify()
            except Exception:
                self._log("  Not a valid image — next variant...")
                image_path.unlink(missing_ok=True)
                return self._generate_pollinations(prompt, filename, model_index + 1,
                                                   search_query, slide_index, seed)

            self._log(f"  [green]Pollinations OK ({model}) → {filename}[/green]")
            return image_path

        except requests.exceptions.Timeout:
            self._log(f"  Timeout ({model}) — next variant...")
            return self._generate_pollinations(prompt, filename, model_index + 1,
                                               search_query, slide_index, seed)
        except Exception as e:
            self._log(f"  {model} error: {str(e)[:80]} — next variant...")
            return self._generate_pollinations(prompt, filename, model_index + 1,
                                               search_query, slide_index, seed)

    def _generate_openverse(self, filename: str, search_query: str = None,
                            slide_index: int = 0) -> Path | None:
        try:
            q = (search_query or "news").strip()[:60]
            self._log(f"  Openverse search: \"{q}\" (slide {slide_index + 1})")
            r = requests.get(
                f"https://api.openverse.org/v1/images/?q={urllib.parse.quote(q)}&page_size=10",
                timeout=20,
            )
            if r.status_code != 200:
                self._log(f"  Openverse HTTP {r.status_code} — gradient fallback")
                return None

            results = r.json().get("results", [])
            if not results:
                self._log(f"  Openverse: no results for \"{q}\" — gradient fallback")
                return None

            pick    = results[slide_index % len(results)]
            img_url = pick.get("url")
            if not img_url:
                return None

            r2 = requests.get(img_url, timeout=40)
            if r2.status_code != 200:
                return None

            image_path = IMAGES_DIR / filename
            image_path.write_bytes(r2.content)

            from PIL import Image as PILImage
            try:
                with PILImage.open(str(image_path)) as img:
                    img.verify()
            except Exception:
                image_path.unlink(missing_ok=True)
                self._log("  Openverse: invalid image — gradient fallback")
                return None

            self._log(f"  [green]Openverse OK → {filename}[/green]")
            return image_path

        except Exception as e:
            self._log(f"  Openverse error: {str(e)[:80]} — gradient fallback")
            return None
