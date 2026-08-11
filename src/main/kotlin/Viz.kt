/*
 * Viz.kt — attention-weight visualizer.
 *
 * Runs one forward pass with an AttnSink attached, capturing the softmaxed
 * attention matrix for every (layer, head). Row i = the token doing the looking
 * ("query"); column j = the token being looked at ("key"). Because of the causal
 * mask, every matrix is lower-triangular: a token can only attend to itself and
 * the past. Brighter cell = more attention.
 *
 * Output: ASCII heatmaps in the terminal, plus a self-contained HTML file with
 * colored grids you can open in a browser.
 */

import java.io.File

/** Human-readable label for a single token id (whitespace made visible). */
private fun tokenLabel(tok: Tokenizer, id: Int): String {
    val s = tok.decode(intArrayOf(id))
    return when {
        s == " " -> "·"
        s == "\n" -> "⏎"
        s.isEmpty() -> "∅"
        else -> s.replace(" ", "·").replace("\n", "⏎")
    }
}

/** Map a weight in [0,1] to a 10-level ASCII shade. */
private fun shade(w: Double): Char {
    val ramp = " .:-=+*#%@"
    var k = (w * (ramp.length - 1)).toInt()
    if (k < 0) k = 0; if (k >= ramp.length) k = ramp.length - 1
    return ramp[k]
}

fun visualizeAttention(model: GPT, tok: Tokenizer, text: String, htmlPath: String? = null) {
    val ids = tok.encode(text).let {
        if (it.size <= model.cfg.blockSize) it else it.copyOfRange(0, model.cfg.blockSize)
    }
    val labels = ids.map { tokenLabel(tok, it) }
    val t = ids.size

    val sink = AttnSink()
    model.forward(ids, sink)   // populates sink.records with one T x T matrix per (layer, head)

    println("\nProbe: \"$text\"  (${t} tokens)")
    println("Rows = query token (looking) · Columns = key token (looked at). Lower-triangular = causal.\n")

    // ---- ASCII heatmaps ----
    for ((layer, head, m) in sink.records) {
        println("Layer $layer · Head $head")
        // Column header: first char of each token label.
        val colHdr = StringBuilder("        ")
        for (j in 0 until t) colHdr.append(labels[j].first()).append(' ')
        println(colHdr)
        for (i in 0 until t) {
            val row = StringBuilder()
            row.append("%6s".format(labels[i].take(6))).append(" |")
            for (j in 0 until t) row.append(shade(m[i][j])).append(' ')
            println(row)
        }
        println()
    }
    println("Shade key: ' '=0  '.:-=+*#'  '@'=1.0\n")

    // ---- HTML heatmaps ----
    if (htmlPath != null) {
        val html = renderHtml(sink, labels, t)
        File(htmlPath).writeText(html)
        println("Wrote ${sink.records.size} attention grids to ${File(htmlPath).absolutePath}")
        println("Open it in a browser to see the colored heatmaps.")
    }
}

private fun renderHtml(sink: AttnSink, labels: List<String>, t: Int): String {
    val sb = StringBuilder()
    sb.append(
        """
        <!doctype html><html><head><meta charset="utf-8"><title>Attention weights</title>
        <style>
          body{font-family:system-ui,sans-serif;background:#0f1115;color:#e6e6e6;margin:24px}
          h1{font-size:18px} h2{font-size:14px;margin:24px 0 8px}
          .grid{border-collapse:collapse;margin-bottom:8px}
          .grid td,.grid th{width:26px;height:26px;text-align:center;font-size:11px;
             border:1px solid #22252c;padding:0}
          .grid th{color:#9aa0aa;font-weight:500}
          .wrap{display:flex;flex-wrap:wrap;gap:24px}
          .cap{color:#9aa0aa;font-size:12px;max-width:640px;line-height:1.5}
        </style></head><body>
        <h1>Self-attention weights</h1>
        <p class="cap">Row = query token (doing the looking). Column = key token (being looked at).
        Cells are lower-triangular because the causal mask forbids attending to the future.
        Brighter = more attention weight (each row sums to 1).</p>
        <div class="wrap">
        """.trimIndent()
    )
    for ((layer, head, m) in sink.records) {
        sb.append("<div><h2>Layer $layer · Head $head</h2><table class=\"grid\"><tr><th></th>")
        for (j in 0 until t) sb.append("<th>${esc(labels[j])}</th>")
        sb.append("</tr>")
        for (i in 0 until t) {
            sb.append("<tr><th>${esc(labels[i])}</th>")
            for (j in 0 until t) {
                val w = m[i][j]
                // Blue-cyan intensity by weight.
                val bg = "rgba(80,170,255,%.3f)".format(w)
                val title = "${esc(labels[i])}→${esc(labels[j])}: %.3f".format(w)
                sb.append("<td style=\"background:$bg\" title=\"$title\"></td>")
            }
            sb.append("</tr>")
        }
        sb.append("</table></div>")
    }
    sb.append("</div></body></html>")
    return sb.toString()
}

private fun esc(s: String) = s
    .replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
