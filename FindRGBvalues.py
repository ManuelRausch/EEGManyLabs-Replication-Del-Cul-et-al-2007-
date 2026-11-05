from psychopy import visual, event, core

# ----------------------------
# Settings
# ----------------------------
WIN_SIZE = (1920, 1080)
RECT_SIZE = (1.6, 1.2)   # in norm units
KEY_STEP = 0.01          # step in PsychoPy rgb units (-1..1). Try 0.005 for finer.
KEY_STEP_FAST = 0.05     # when holding SHIFT

# ----------------------------
# Window
# ----------------------------
win = visual.Window(
    size=WIN_SIZE,
    units="norm",
    color=[-1, -1, -1],
    colorSpace="rgb", 
    fullscr=True
)

# ----------------------------
# Stimuli
# ----------------------------
rect = visual.Rect(
    win,
    width=RECT_SIZE[0],
    height=RECT_SIZE[1],
    fillColor=[0, 0, 0],
    lineColor=None,
    colorSpace="rgb"
)

slider = visual.Slider(
    win,
    pos=(0, -0.75),
    size=(1.4, 0.08),
    ticks=[-1, -0.5, 0, 0.5, 1],
    granularity=0,            # continuous
    style=["rating"],
    labelHeight=0.05, 
    color = "dodgerblue"
)
slider.reset()
slider.markerPos = 0.0        # start in the middle

info = visual.TextStim(
    win,
    pos=(0, 0.85),
    height=0.05,
    color="white",
    text=""
)

help_text = visual.TextStim(
    win,
    pos=(0, -0.92),
    height=0.045,
    color="white",
    text="Mouse or arrows (←/→) to adjust. SHIFT+arrows = bigger steps. SPACE accept. ESC quit."
)

# ----------------------------
# Helpers
# ----------------------------
def clamp(x, lo=-1.0, hi=1.0):
    return max(lo, min(hi, x))

def rgb_to_255(x_rgb):
    v = int(round((x_rgb + 1) * 0.5 * 255))
    return max(0, min(255, v))

# ----------------------------
# Loop
# ----------------------------
accepted_value = None

while True:
    keys = event.getKeys()

    if "escape" in keys:
        break

    if "space" in keys:
        accepted_value = slider.getRating()
        if accepted_value is None:
            # If not "rated" yet, fall back to markerPos
            accepted_value = slider.markerPos if slider.markerPos is not None else 0.0
        break

    # Determine current value (mouse might update rating; otherwise use markerPos)
    current = slider.getRating()
    if current is None:
        current = slider.markerPos if slider.markerPos is not None else 0.0

    # Keyboard nudging
    step = KEY_STEP_FAST if ("lshift" in keys or "rshift" in keys) else KEY_STEP
    if "left" in keys or "a" in keys:
        current -= step
    if "right" in keys or "d" in keys:
        current += step

    # Apply update back to slider
    current = clamp(current, -1.0, 1.0)
    slider.markerPos = current
    # Note: we intentionally do NOT call slider.reset() here (would wipe interaction)

    # Update rectangle + text
    rect.fillColor = [current, current, current]
    val255 = rgb_to_255(current)
    info.text = (
        f"Current grayscale:\n"
        f"PsychoPy rgb (-1..1):  [{current:+.3f}, {current:+.3f}, {current:+.3f}]\n"
        f"8-bit (0..255):        [{val255:3d}, {val255:3d}, {val255:3d}]"
    )

    rect.draw()
    slider.draw()
    info.draw()
    help_text.draw()
    win.flip()

win.close()
core.quit()

if accepted_value is not None:
    val255 = rgb_to_255(accepted_value)
    print("\nAccepted value:")
    print(f"  PsychoPy rgb (-1..1): [{accepted_value:+.3f}, {accepted_value:+.3f}, {accepted_value:+.3f}]")
    print(f"  8-bit (0..255):       [{val255:3d}, {val255:3d}, {val255:3d}]")
