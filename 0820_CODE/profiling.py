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


def profile_axis(
    user_history, factor_summaries, far_items, close_items, method: str = "svd", axis_labels=None
):
    """Axis-based: profile generated conditioned on matrix-factorization (method) axes, plus
    movies the axes already explain well (close) contrasted with ones they don't (far).
    axis_labels lets interpretable axes (e.g. genre names) be shown by name instead of "Axis N"."""
    factors_text = "\n".join(
        f"{axis_labels[i] if axis_labels else f'Axis {i + 1}'}: "
        + ", ".join(f"{t} ({'+' if w >= 0 else '-'})" for t, w in fs[:5])
        for i, fs in enumerate(factor_summaries)
    )
    close_text = "\n".join(
        f"- {t}: matches the axes' prediction closely ({e:+.2f})" for t, e in close_items
    ) or "(none)"
    far_text = "\n".join(
        f"- {t}: rated {'higher' if e > 0 else 'lower'} than the axes predict ({e:+.2f})"
        for t, e in far_items
    ) or "(none)"
    lines = "\n".join(f"- {t} ({g}) rated {r}/5" for t, g, r in user_history)
    prompt = (
        f"Below are the user's latent taste axes extracted via {method.upper()}, "
        "this user's watch history, and how well those axes explain specific movies the user rated.\n\n"
        "[Latent axes] Each axis is a spectrum: '+' movies define one end, '-' movies the "
        f"opposite end (e.g. liking one end often means disliking the other).\n{factors_text}\n\n"
        f"[Watch history]\n{lines}\n\n"
        f"[Well explained by existing axes]\n{close_text}\n\n"
        "[Not well explained by existing axes] A positive gap means the user secretly likes "
        f"it more than these axes suggest; a negative gap means they like it less.\n{far_text}\n\n"
        "Using the contrast between what the axes already capture and what they miss, "
        "point out the taste characteristics the existing axes are missing, and "
        "describe the user's movie taste in about 200 characters."
    )
    return chat([{"role": "user", "content": prompt}])
