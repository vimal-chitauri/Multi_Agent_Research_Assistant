import os
import json
from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel

load_dotenv()
console = Console()

NICHE = os.getenv("NICHE", "technology")


def run_pipeline():
    from agents.trend_agent import TrendAgent
    from agents.content_agent import ContentAgent
    from agents.image_agent import ImageAgent
    from agents.music_agent import MusicAgent
    from agents.video_agent import VideoAgent
    from agents.verifier_agent import VerifierAgent
    from agents.publisher_agent import PublisherAgent

    console.print(Panel(
        f"[bold]Multi-Agent Marketing System[/bold]\n"
        f"Niche: [green]{NICHE}[/green]  |  Agents: TrendAgent → ContentAgent",
        title="Pipeline Starting"
    ))

    console.print("\n[bold yellow]── AGENT 1: TREND RESEARCH ──[/bold yellow]")
    trend_agent = TrendAgent(niche=NICHE)
    trend_result = trend_agent.run()

    if not trend_result.success:
        console.print(f"[red]TrendAgent failed: {trend_result.errors}[/red]")
        return

    console.print(f"[green]✓ Found {len(trend_result.data['trends'])} trending topics[/green]")
    for t in trend_result.data["trends"]:
        console.print(f"  [{t['rank']}] {t['topic']}  ({t['emotion']} / {t['post_type']})")

    console.print("\n[bold yellow]── AGENT 2: CONTENT CREATION ──[/bold yellow]")
    content_agent = ContentAgent(niche=NICHE)
    content_result = content_agent.run({
        "trends": trend_result.data["trends"],
        "niche": NICHE,
    })

    if not content_result.success:
        console.print(f"[red]ContentAgent failed: {content_result.errors}[/red]")
        return

    console.print(f"\n[green]✓ Generated {content_result.data['total']} posts[/green]\n")

    for post in content_result.data["posts"]:
        console.print(Panel(
            f"[bold cyan]TOPIC:[/bold cyan] {post['topic']}\n\n"
            f"[bold]TWITTER ({post['twitter']['char_count']} chars):[/bold]\n"
            f"{post['twitter']['text']}\n\n"
            f"[bold]INSTAGRAM HOOK:[/bold]\n"
            f"{post['instagram']['hook']}\n\n"
            f"[bold]INSTAGRAM CAPTION:[/bold]\n"
            f"{post['instagram']['caption']}\n\n"
            f"[bold]HASHTAGS:[/bold] {' '.join(['#' + h for h in post['instagram']['hashtags'][:8]])}\n\n"
            f"[bold]IMAGE PROMPT:[/bold]\n{post['image_prompt']}\n\n"
            f"[bold]BEST TIME TO POST:[/bold] {post['best_post_time']}\n"
            f"[bold]WHY IT WORKS:[/bold] {post['why_this_works']}",
            title=f"[green]Post #{post['rank']}[/green]",
            border_style="green"
        ))

    console.print("\n[bold yellow]── AGENT 3: IMAGE GENERATION ──[/bold yellow]")
    image_agent  = ImageAgent()
    image_result = image_agent.run({"posts": content_result.data["posts"]})

    if not image_result.success:
        console.print(f"[red]ImageAgent failed: {image_result.errors}[/red]")
        return

    console.print(f"[green]✓ {image_result.data['images_generated']}/{image_result.data['total']} images generated[/green]")

    console.print("\n[bold yellow]── AGENT 4: MUSIC SELECTION ──[/bold yellow]")
    music_agent  = MusicAgent(niche=NICHE)
    music_result = music_agent.run({"posts": image_result.data["posts"], "niche": NICHE})

    if not music_result.success:
        console.print(f"[yellow]MusicAgent warning: {music_result.errors} — continuing without music[/yellow]")
        music_posts = image_result.data["posts"]
    else:
        console.print(f"[green]✓ {music_result.data['music_added']}/{music_result.data['total']} tracks fetched[/green]")
        music_posts = music_result.data["posts"]

    console.print("\n[bold yellow]── AGENT 5: VIDEO CREATION ──[/bold yellow]")
    video_agent  = VideoAgent(duration=30, niche=NICHE)
    video_result = video_agent.run({"posts": music_posts, "niche": NICHE})

    if not video_result.success:
        console.print(f"[red]VideoAgent failed: {video_result.errors}[/red]")
        return

    console.print(f"[green]✓ {video_result.data['videos_created']}/{video_result.data['total']} videos created[/green]")

    console.print("\n[bold yellow]── AGENT 6: VERIFICATION + APPROVAL ──[/bold yellow]")
    verifier_agent = VerifierAgent(niche=NICHE)
    verify_result = verifier_agent.run({
        "posts": video_result.data["posts"],
    })

    if not verify_result.success:
        console.print(f"[red]No posts approved: {verify_result.errors}[/red]")
        return

    console.print(f"\n[bold green]✓ {verify_result.data['total']} post(s) approved and ready to publish.[/bold green]")

    console.print("\n[bold yellow]── AGENT 7: INSTAGRAM PUBLISHER ──[/bold yellow]")
    console.print("[dim]Running in DRY RUN mode — set dry_run=False to post live[/dim]\n")

    publisher = PublisherAgent(dry_run=True)
    publish_result = publisher.run({
        "posts": verify_result.data["posts"],
    })

    if publish_result.success:
        total = publish_result.data["total"]
        mode  = "DRY RUN" if publish_result.data["dry_run"] else "LIVE"
        console.print(f"\n[bold green]✓ {total} post(s) processed [{mode}][/bold green]")
    else:
        console.print(f"[red]Publisher failed: {publish_result.errors}[/red]")

    console.print("\n[bold green]Pipeline complete. All 6 agents ran successfully.[/bold green]")
    return publish_result


if __name__ == "__main__":
    run_pipeline()
