#!/usr/bin/env python3
"""Fill Submission_Final_forGithub.pptx with Track-3 content."""
from __future__ import annotations

from pathlib import Path

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.util import Emu, Inches, Pt

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "Submission_Final_forGithub.pptx"
OUT = ROOT / "Submission_Final_forGithub.pptx"
# Fallback when the main file is locked open in PowerPoint.
OUT_ALT = ROOT / "Submission_Final_readable.pptx"
COPY = ROOT / "submission" / "SRMIST KTR_2 fast 2 curious.pptx"

# Slide background is near-black (#0A0E1A) — keep all readable text light.
NAVY = RGBColor(0xE8, 0xEE, 0xF8)   # light headings (was dark navy)
TEAL = RGBColor(0x5E, 0xE7, 0xF0)   # bright cyan accent
BODY = RGBColor(0xF2, 0xF5, 0xFA)   # primary body
MUTED = RGBColor(0xB0, 0xBC, 0xD0)  # secondary / captions
WHITE = RGBColor(0xFF, 0xFF, 0xFF)


def _set_runs(shape, lines: list[tuple[str, bool, int, RGBColor]], *, align=PP_ALIGN.LEFT):
    tf = shape.text_frame
    tf.clear()
    tf.word_wrap = True
    first = True
    for text, bold, size, color in lines:
        p = tf.paragraphs[0] if first else tf.add_paragraph()
        first = False
        p.alignment = align
        p.space_after = Pt(4)
        run = p.add_run()
        run.text = text
        run.font.bold = bold
        run.font.size = Pt(size)
        run.font.color.rgb = color
        run.font.name = "Calibri"


def _replace_title(shape, text: str, *, size: int = 28, color: RGBColor = WHITE):
    tf = shape.text_frame
    tf.clear()
    p = tf.paragraphs[0]
    p.alignment = PP_ALIGN.LEFT
    run = p.add_run()
    run.text = text
    run.font.bold = True
    run.font.size = Pt(size)
    run.font.color.rgb = color
    run.font.name = "Calibri"


def _add_body(slide, left, top, width, height, blocks: list[tuple[str, bool, int, RGBColor]]):
    box = slide.shapes.add_textbox(left, top, width, height)
    _set_runs(box, blocks)
    return box


def _strip_textboxes(slide) -> None:
    """Remove previously injected text boxes so fills do not stack/duplicate."""
    for sh in list(slide.shapes):
        if sh.has_text_frame and sh.name.startswith("TextBox"):
            el = sh._element
            el.getparent().remove(el)
        elif sh.has_text_frame and getattr(sh, "shape_type", None) is not None:
            # python-pptx: TEXT_BOX == 17
            try:
                if int(sh.shape_type) == 17:
                    el = sh._element
                    el.getparent().remove(el)
            except Exception:
                pass


def _recolor_all_text(slide) -> None:
    """Force every run on the slide to a light-on-dark palette."""
    for sh in slide.shapes:
        if not sh.has_text_frame:
            continue
        for p in sh.text_frame.paragraphs:
            for r in p.runs:
                if not r.text:
                    continue
                bold = bool(r.font.bold)
                size_pt = (r.font.size.pt if r.font.size else 14)
                if size_pt >= 28:
                    r.font.color.rgb = WHITE
                elif bold and size_pt >= 15:
                    r.font.color.rgb = TEAL
                elif bold:
                    r.font.color.rgb = NAVY
                else:
                    r.font.color.rgb = BODY if size_pt >= 12 else MUTED


def fill_title(slide) -> None:
    _strip_textboxes(slide)
    # Track No badge text
    for sh in slide.shapes:
        if not sh.has_text_frame:
            continue
        raw = sh.text_frame.text.strip()
        if raw in ("Track No", "Track 03") or raw.startswith("Track 03"):
            if "Path Finder" not in raw and len(raw) < 20:
                _replace_title(sh, "Track 03", size=18, color=WHITE)
        elif raw.startswith("XR Hackathon") or raw.startswith("Path Finder"):
            _set_runs(
                sh,
                [
                    ("XR Hackathon 2026", True, 36, WHITE),
                    ("Path Finder — IMU-only 6-DoF", True, 22, TEAL),
                    ("Never below the stay-put floor · win on rotation & efficiency", False, 14, MUTED),
                ],
            )
        elif "Team" in raw and ("College" in raw or "TeamName" in raw or "Team Name" in raw):
            _set_runs(
                sh,
                [
                    ("Team: 2 fast 2 curious    College: SRMIST KTR", True, 18, WHITE),
                    ("Submission: SRMIST KTR_2 fast 2 curious", False, 12, MUTED),
                ],
            )
        elif raw.startswith("Members"):
            _set_runs(
                sh,
                [
                    ("Members", True, 14, TEAL),
                    ("[Name 1]  ·  [Name 2]", False, 13, BODY),
                    ("[Name 3]  ·  [Name 4]", False, 13, BODY),
                ],
            )
    _recolor_all_text(slide)


def fill_problem(slide) -> None:
    _strip_textboxes(slide)
    _replace_title(slide.shapes[2], "THE PROBLEM")
    _add_body(
        slide,
        Inches(0.45),
        Inches(1.05),
        Inches(9.1),
        Inches(4.2),
        [
            ("Track 03 — Path Finder (GPS-denied / jammed IMU)", True, 16, TEAL),
            (
                "Estimate a full 6-DoF trajectory from a degraded IMU stream only — "
                "no vision, no GPS, no waypoints at runtime.",
                False,
                14,
                BODY,
            ),
            ("", False, 8, BODY),
            ("What the official degrade_imu.py does", True, 15, NAVY),
            ("• Progressive noise + bias drift as the mission advances", False, 13, BODY),
            ("• Frame dropouts (missing rows) and sensor spikes", False, 13, BODY),
            ("• Last ~5 s: IMU values unusable → must dead-reckon / hold", False, 13, BODY),
            ("", False, 8, BODY),
            ("Budgets (FAQ)", True, 15, NAVY),
            ("• ≤ 8 MB algorithm heap   ·   ≤ 20 ms compute per 1 s of IMU", False, 13, BODY),
            ("• Single CPU thread · offline · Python main.py only", False, 13, BODY),
            ("", False, 8, BODY),
            ("Scoring (100)", True, 15, NAVY),
            (
                "Efficiency 40% (mem 20 + runtime 20)  ·  "
                "Trajectory 60% (waypoints 30 · blind-5s 20 · dropout 10)",
                False,
                13,
                BODY,
            ),
        ],
    )


def fill_solution(slide) -> None:
    _strip_textboxes(slide)
    _replace_title(slide.shapes[2], "SOLUTION")
    _add_body(
        slide,
        Inches(0.45),
        Inches(1.05),
        Inches(9.1),
        Inches(4.2),
        [
            ("Stay-put floor first — then win where the IMU is trustworthy", True, 16, TEAL),
            (
                "A naive double-integrator drifts kilometres under degrade. "
                "Doing nothing (stay-put) sets a hard ATE floor. We hug that floor "
                "on translation and spend IMU signal on rotation.",
                False,
                13,
                BODY,
            ),
            ("", False, 6, BODY),
            ("Core design", True, 15, NAVY),
            ("1. Per-file gyro/accel bias + protect-window gravity align (no EuRoC hard-codes)", False, 13, BODY),
            ("2. Always integrate translation; cascaded 2nd-order velocity HP (τ = 0.35 s)", False, 13, BODY),
            ("3. Attitude at ~50 Hz keyframes (block-average lean) + upsample to every row", False, 13, BODY),
            ("4. Protect-window Mahony only — mid-flight gravity aid rejected", False, 13, BODY),
            ("5. Blind last ~5 s: hold pose (1 output row per IMU row)", False, 13, BODY),
            ("6. Exception-safe hold-pose fallback if anything crashes", False, 13, BODY),
            ("", False, 6, BODY),
            (
                "USP: compliant efficiency + strong RPE-rot + never below stay-put on translation",
                True,
                13,
                TEAL,
            ),
        ],
    )


def fill_architecture(slide) -> None:
    _strip_textboxes(slide)
    _replace_title(slide.shapes[2], "ARCHITECTURE")
    _add_body(
        slide,
        Inches(0.4),
        Inches(1.0),
        Inches(9.2),
        Inches(4.3),
        [
            ("main.py  →  estimator.run_estimator  (dual-pass: untraced time + traced heap)", True, 13, TEAL),
            ("", False, 4, BODY),
            ("load_imu_csv          chunked load → float32 memmap sensors (spill ≠ CWD)", False, 12, BODY),
            ("analyze_imu           spikes / stuck / gaps / blind window / dropouts", False, 12, BODY),
            ("run_lean_ins", True, 12, NAVY),
            ("   ├─ bias + q0       time-horizon bias + gravity align", False, 12, BODY),
            ("   ├─ protect Mahony  ~2 s attitude quality only", False, 12, BODY),
            ("   ├─ att_scan        vectorized post-protect gyro attitude (~50 Hz)", False, 12, BODY),
            ("   └─ hp_block        cascaded 2nd-order velocity high-pass", False, 12, BODY),
            ("write TUM             keyframe upsample + tum_bytes (no savetxt)", False, 12, BODY),
            ("", False, 6, BODY),
            ("Stack", True, 13, NAVY),
            ("Python 3 · NumPy only in measured path · no SciPy · Docker + one-command gate", False, 12, BODY),
            ("", False, 4, BODY),
            ("Output: estimated_trajectory.txt  (t tx ty tz qx qy qz qw) + metadata.json", False, 12, MUTED),
        ],
    )


def fill_results(slide) -> None:
    _strip_textboxes(slide)
    _replace_title(slide.shapes[2], "RESULTS & TRACK OUTPUT")
    _add_body(
        slide,
        Inches(0.4),
        Inches(1.0),
        Inches(9.2),
        Inches(4.3),
        [
            ("Regression gate (official degrade, 3 seeds × 5 MH) — 15 / 15 PASS", True, 15, TEAL),
            ("", False, 4, BODY),
            ("Translation   ATE / RPE-trans / blind-5s  all ≤ stay-put gates (hug the floor)", False, 12, BODY),
            ("Rotation      RPE-rot ~0.4–1.2°   gains ~7–25× vs held attitude", False, 12, BODY),
            ("MH01 ex.      ATE 4.29 m vs SP 4.32 m · RPE-rot 0.54° (gain 16.5×)", False, 12, BODY),
            ("", False, 6, BODY),
            ("Efficiency (FAQ)", True, 14, NAVY),
            ("Runtime   ~2–3.5 ms / s IMU untraced   ·   ~13–16 ms/s under tracemalloc (≤20)", False, 12, BODY),
            ("Heap      tracemalloc ~2–3.5 MB on MH01 (≤8 MB algorithm heap)", False, 12, BODY),
            ("", False, 6, BODY),
            ("Submission bundle (Slide 17)", True, 14, NAVY),
            ("SRMIST KTR_2 fast 2 curious.zip → estimated_trajectory.txt · metadata.json · src/main.py", False, 12, BODY),
            ("Evaluated with evo (Umeyama ATE / RPE) against hidden GT", False, 12, MUTED),
        ],
    )


def fill_diff(slide) -> None:
    _strip_textboxes(slide)
    _replace_title(slide.shapes[2], "DIFFERENTIATION")
    _add_body(
        slide,
        Inches(0.4),
        Inches(1.0),
        Inches(9.2),
        Inches(4.3),
        [
            ("Why this wins points under the official rubric", True, 15, TEAL),
            ("", False, 4, BODY),
            ("Floor-aware translation", True, 13, NAVY),
            ("We treat stay-put as the information-theoretic floor under jammed IMU — "
             "not a bug. Cascaded 2nd-order HP keeps ATE within 5% of that floor.", False, 12, BODY),
            ("", False, 4, BODY),
            ("Rotation where signal exists", True, 13, NAVY),
            ("Gyro integration at locked ~50 Hz block-average lean preserves RPE-rot gains; "
             "full-rate attitude collapsed the gate (kept out on purpose).", False, 12, BODY),
            ("", False, 4, BODY),
            ("Efficiency by construction", True, 13, NAVY),
            ("Float32 memmap · vectorized att_scan/hp_block · tum_bytes · 30 s bias horizon · "
             "dual-pass timing so FAQ heap and wall-clock are both honest.", False, 12, BODY),
            ("", False, 4, BODY),
            ("Grader-safe I/O", True, 13, NAVY),
            ("RO-cwd safe spill · RAM fallback · exact 1 pose / IMU row · hold-pose on crash", False, 12, BODY),
        ],
    )


def fill_demo(slide) -> None:
    _strip_textboxes(slide)
    _replace_title(slide.shapes[2], "DEMO & LOCAL SETUP")
    _add_body(
        slide,
        Inches(0.4),
        Inches(1.0),
        Inches(9.2),
        Inches(4.3),
        [
            ("Setup", True, 14, TEAL),
            ("pip install -r requirements.txt", False, 12, BODY),
            ("docker build -t track3-pathfinder .   &&   docker run --rm track3-pathfinder", False, 12, BODY),
            ("", False, 4, BODY),
            ("Run estimator", True, 14, TEAL),
            ("python src/main.py <degraded_imu.csv> --out estimated_trajectory.txt --meta metadata.json", False, 11, BODY),
            ("python tools/validate_output.py <imu.csv> estimated_trajectory.txt", False, 11, BODY),
            ("", False, 4, BODY),
            ("Reproduce gate / pack", True, 14, TEAL),
            ("python tools/regression_gate.py --seeds 42,7,99          # expect 15/15", False, 11, BODY),
            ("python tools/pack_submission.py --team \"SRMIST KTR_2 fast 2 curious\" \\", False, 11, BODY),
            ("    --traj outputs/MH_01_easy/estimated_trajectory.txt --meta outputs/MH_01_easy/metadata.json", False, 11, BODY),
            ("", False, 4, BODY),
            ("Inputs accepted: CSV · folder with imu0/data.csv · zip of that layout", False, 12, MUTED),
        ],
    )


def fill_brownie(slide) -> None:
    _strip_textboxes(slide)
    _replace_title(slide.shapes[2], "LEARNINGS & NEGATIVE RESULTS")
    _add_body(
        slide,
        Inches(0.4),
        Inches(1.0),
        Inches(9.2),
        Inches(4.3),
        [
            ("What we tried and rejected (kept out of the freeze)", True, 14, TEAL),
            ("• Hover / |a|≈g gates + ZUPT — killed real motion, lost blind-5s", False, 12, BODY),
            ("• 1st-order velocity HP — residual bias → km-scale walk", False, 12, BODY),
            ("• Mid-flight Mahony / gravity aid — worsened ATE-rot / RPE-rot", False, 12, BODY),
            ("• Full-rate attitude — collapsed rotation gains (gate 0/5)", False, 12, BODY),
            ("• Blind constant-ω coast — hold pose wins the last 5 s", False, 12, BODY),
            ("", False, 6, BODY),
            ("Honest limitation", True, 14, NAVY),
            (
                "Translation ties stay-put by design. Under jammed IMU, chasing lower ATE "
                "usually means trusting accel more — and that is what explodes. We compete "
                "on rotation, blind-window discipline, dropout handling, and efficiency.",
                False,
                12,
                BODY,
            ),
        ],
    )


def fill_thanks(slide) -> None:
    _strip_textboxes(slide)
    for sh in slide.shapes:
        if sh.has_text_frame and "Thank You" in sh.text_frame.text:
            _set_runs(
                sh,
                [
                    ("Thank You", True, 40, WHITE),
                ],
                align=PP_ALIGN.CENTER,
            )
    _add_body(
        slide,
        Inches(1.5),
        Inches(3.35),
        Inches(7.0),
        Inches(1.2),
        [
            ("Track 03 · Path Finder", True, 16, TEAL),
            ("SRMIST KTR  ·  2 fast 2 curious  ·  Samsung XR Hackathon 2026", False, 13, MUTED),
            ("Questions?  prism@samsung.com", False, 12, MUTED),
        ],
    )
    # center the thanks block paragraphs
    for sh in slide.shapes:
        if sh.has_text_frame and "Track 03" in sh.text_frame.text:
            for p in sh.text_frame.paragraphs:
                p.alignment = PP_ALIGN.CENTER
    _recolor_all_text(slide)


def main() -> int:
    # Prefer unlocked copies first (original is often open in PowerPoint).
    src = next((p for p in (OUT_ALT, COPY, SRC) if p.exists()), SRC)
    prs = Presentation(str(src))
    print(f"loaded {src}")

    fill_title(prs.slides[0])
    fill_problem(prs.slides[1])
    fill_solution(prs.slides[2])
    fill_architecture(prs.slides[3])
    fill_results(prs.slides[4])
    fill_diff(prs.slides[5])
    fill_demo(prs.slides[6])
    fill_brownie(prs.slides[7])
    fill_thanks(prs.slides[8])
    for slide in prs.slides:
        _recolor_all_text(slide)
    COPY.parent.mkdir(parents=True, exist_ok=True)
    saved = []
    for path in (OUT, COPY, OUT_ALT):
        try:
            prs.save(str(path))
            saved.append(path)
        except PermissionError:
            print(f"locked, skipped: {path}")
    if not saved:
        raise SystemExit(
            "Could not save any PPTX — close PowerPoint (and delete ~$ lock files) then re-run."
        )
    for path in saved:
        print(f"wrote {path}")
    print("Team set to SRMIST KTR_2 fast 2 curious — fill member names on slide 1 if needed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
