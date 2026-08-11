from llm_client import chat


def profile_raw(user_history):
    """Baseline: LLM writes a profile directly from watch history alone (KAR/ONCE style)."""
    lines = "\n".join(f"- {t} ({g}) rated {r}/5" for t, g, r in user_history)
    prompt = (
        "Below is a list of movies a user has watched and rated.\n"
        f"{lines}\n\n"
        "Based on this, describe the user's movie taste in about 200 characters. "
        "Be specific about genres, tone, and preference patterns."
    )
    return chat([{"role": "user", "content": prompt}])


def profile_axis(user_history, factor_summaries, residual_items, method: str = "svd"):
    """Axis-based: profile generated conditioned on matrix-factorization (method) axes + residuals."""
    factors_text = "\n".join(
        f"Axis {i + 1}: "
        + ", ".join(f"{t} ({'+' if w >= 0 else '-'})" for t, w in fs[:5])
        for i, fs in enumerate(factor_summaries)
    )
    residual_text = "\n".join(
        f"- {t}: rated {'higher' if e > 0 else 'lower'} than the axes predict ({e:+.2f})"
        for t, e in residual_items
    ) or "(none)"
    lines = "\n".join(f"- {t} ({g}) rated {r}/5" for t, g, r in user_history)
    prompt = (
        f"Below are the user's latent taste axes extracted via {method.upper()}, "
        "this user's watch history, and movies that the existing axes fail to explain well.\n\n"
        "[Latent axes] Each axis is a spectrum: '+' movies define one end, '-' movies the "
        f"opposite end (e.g. liking one end often means disliking the other).\n{factors_text}\n\n"
        f"[Watch history]\n{lines}\n\n"
        "[Movies not explained by existing axes] A positive gap means the user secretly likes "
        f"it more than these axes suggest; a negative gap means they like it less.\n{residual_text}\n\n"
        "Point out the taste characteristics that the existing axes are missing, and "
        "describe the user's movie taste in about 200 characters."
    )
    return chat([{"role": "user", "content": prompt}])
