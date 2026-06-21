import os
import re
import time
import requests
import urllib.parse
from pathlib import Path
from dotenv import load_dotenv
from agents.base_agent import BaseAgent, AgentResult
from huggingface_hub import InferenceClient

load_dotenv(override=True)

IMAGES_DIR = Path(__file__).parent.parent / "generated_images"
IMAGES_DIR.mkdir(exist_ok=True)

HF_MODELS = [
    "black-forest-labs/FLUX.1-schnell",
    "stabilityai/stable-diffusion-xl-base-1.0",
    "stabilityai/stable-diffusion-2-1",
    "runwayml/stable-diffusion-v1-5",
    "dreamlike-art/dreamlike-photoreal-2.0",
    "prompthero/openjourney",
]

POLLINATIONS_MODELS = [
    "flux",
    "flux-realism",
    "flux-anime",
    "flux-3d",
    "turbo",
    "dreamshaper",
]


class ImageAgent(BaseAgent):

    def __init__(self):
        super().__init__(name="ImageAgent", model="fast")
        self.hf_token = os.getenv("HUGGINGFACE_TOKEN")

    @property
    def system_prompt(self) -> str:
        return "You are an image prompt optimizer for social media posts."

    def run(self, input_data: dict) -> AgentResult:
        posts = input_data.get("posts", [])
        if not posts:
            return self._failure("No posts received from ContentAgent.")

        if self.hf_token:
            self._log(
                f"Image chain: [bold]{len(HF_MODELS)} HF[/bold] → "
                f"[bold]{len(POLLINATIONS_MODELS)} Pollinations[/bold] → "
                f"[bold]Openverse[/bold] → gradient"
            )
        else:
            self._log(
                f"[yellow]No HF token — chain: "
                f"{len(POLLINATIONS_MODELS)} Pollinations → Openverse (CC photos) → gradient[/yellow]"
            )

        images_generated = 0

        for post in posts:
            topic = post["topic"]
            self._log(f"Generating image(s) for: [cyan]{topic}[/cyan]")

            if "image_prompts" in post and isinstance(post["image_prompts"], list):
                paths = []
                for idx, raw_prompt in enumerate(post["image_prompts"][:3]):
                    prompt   = self._enhance_prompt(raw_prompt)
                    filename = f"post_{post['rank']}_{int(time.time())}_{idx}.png"
                    path     = self._generate(prompt, filename,
                                              search_query=topic, slide_index=idx)
                    if path:
                        paths.append(str(path))
                        images_generated += 1
                    else:
                        self._log(f"  [red]Slide {idx + 1}: all backends failed[/red]")
                    if idx < 2:
                        time.sleep(4)
                post["image_paths"]  = paths
                post["image_path"]   = paths[0] if paths else None
                post["image_status"] = "generated" if paths else "failed"
            else:
                prompt   = self._enhance_prompt(post.get("image_prompt", topic))
                filename = f"post_{post['rank']}_{int(time.time())}.png"
                path     = self._generate(prompt, filename, search_query=topic)
                if path:
                    post["image_path"]   = str(path)
                    post["image_paths"]  = [str(path)]
                    post["image_status"] = "generated"
                    images_generated    += 1
                else:
                    post["image_path"]   = None
                    post["image_paths"]  = []
                    post["image_status"] = "failed"
                    self._log(f"  [red]All backends failed for: {topic}[/red]")
                if len(posts) > 1:
                    time.sleep(4)

        self._log(f"[green]{images_generated} image(s) generated across {len(posts)} post(s)[/green]")
        return self._success(
            data={"posts": posts, "total": len(posts), "images_generated": images_generated},
            reasoning=f"{images_generated} images via HF + Pollinations + Openverse chain"
        )


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
            "no real people, no faces, symbolic imagery, "
            "professional photography, vibrant colors, high resolution, "
            "1:1 aspect ratio, sharp focus, cinematic lighting, modern aesthetic"
        )


    def _generate(self, prompt: str, filename: str,
                  model_index: int = 0, _rate_retries: int = 0,
                  search_query: str = None, slide_index: int = 0) -> Path | None:
        if not self.hf_token or model_index >= len(HF_MODELS):
            return self._generate_pollinations(prompt, filename, 0,
                                               search_query=search_query,
                                               slide_index=slide_index)

        model = HF_MODELS[model_index]
        try:
            client = InferenceClient(token=self.hf_token)
            self._log(f"  HF [{model_index + 1}/{len(HF_MODELS)}]: {model.split('/')[-1]}")
            image = client.text_to_image(prompt, model=model)
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
                return self._generate(prompt, filename, model_index, _rate_retries, search_query, slide_index)

            elif "429" in err or "rate" in err.lower() or "too many" in err.lower():
                if _rate_retries < 3:
                    wait = 30 * (_rate_retries + 1)
                    self._log(f"  Rate limited — backing off {wait}s ({_rate_retries + 1}/3)...")
                    time.sleep(wait)
                    return self._generate(prompt, filename, model_index, _rate_retries + 1, search_query, slide_index)
                self._log("  Rate limit exhausted — next HF model...")
                return self._generate(prompt, filename, model_index + 1, 0, search_query, slide_index)

            elif "402" in err or "payment" in err.lower():
                self._log("  Pro plan required — next HF model...")
                return self._generate(prompt, filename, model_index + 1, 0, search_query, slide_index)

            elif ("nsfw" in err.lower() or "safety" in err.lower()
                  or "inappropriate" in err.lower() or "400" in err):
                self._log("  Safety filter — next HF model...")
                return self._generate(prompt, filename, model_index + 1, 0, search_query, slide_index)

            else:
                self._log("  Error — next HF model...")
                return self._generate(prompt, filename, model_index + 1, 0, search_query, slide_index)


    def _generate_pollinations(self, prompt: str, filename: str,
                                model_index: int = 0,
                                search_query: str = None,
                                slide_index: int = 0) -> Path | None:
        if model_index >= len(POLLINATIONS_MODELS):
            self._log("  Pollinations all rate-limited — trying Openverse...")
            return self._generate_openverse(filename, search_query=search_query,
                                            slide_index=slide_index)

        model = POLLINATIONS_MODELS[model_index]
        self._log(f"  Pollinations [{model_index + 1}/{len(POLLINATIONS_MODELS)}]: {model}")

        try:
            encoded = urllib.parse.quote(prompt[:480])
            url = (
                f"https://image.pollinations.ai/prompt/{encoded}"
                f"?width=768&height=768&model={model}&seed={int(time.time())}"
            )
            r = requests.get(url, timeout=90, stream=True)

            if r.status_code == 402:
                self._log(f"  Rate-limited ({model}) — next variant...")
                return self._generate_pollinations(prompt, filename, model_index + 1,
                                                   search_query, slide_index)

            if r.status_code != 200:
                self._log(f"  HTTP {r.status_code} — next variant...")
                return self._generate_pollinations(prompt, filename, model_index + 1,
                                                   search_query, slide_index)

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
                                                   search_query, slide_index)

            self._log(f"  [green]Pollinations OK ({model}) → {filename}[/green]")
            return image_path

        except requests.exceptions.Timeout:
            self._log(f"  Timeout ({model}) — next variant...")
            return self._generate_pollinations(prompt, filename, model_index + 1,
                                               search_query, slide_index)
        except Exception as e:
            self._log(f"  {model} error: {str(e)[:80]} — next variant...")
            return self._generate_pollinations(prompt, filename, model_index + 1,
                                               search_query, slide_index)

    
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
