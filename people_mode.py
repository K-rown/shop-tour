"""What the engine does with the OTHER people in a stop's picture: SHOP_TOUR_PEOPLE.

  keep    (the default since 2026-10-30 - the owner, on his road walks: "the blurs work very well at taking users out but they often cover large [areas] the
          viewer could otherwise use... just blur faces") people are left exactly as they were photographed; only the faces, plates and printed names are
          blurred (the pii step); nothing is held.
  blur    people stay in the picture, smudged: the privacy gate still FINDS them (the same detector and SAM masks as ever), but no stop is ever held, the
          paint-out and the privacy margin do not run, and nobody is asked to review anything. Faces, plates and printed names are blurred on top, as always.
  remove  the older behaviour, byte for byte: every other person is painted out (remove_others.py, SHOP_TOUR_REMOVE_OTHERS=0 still turns that off), the gate holds
          a stop where a person is still found, the margin holds what would be held at another camera height, and the review (privacy_gate.py --keep / --leave-out) applies.
In all three modes the person holding the stick (the "capturer") is dealt with by the capturer step: painted out in remove mode, smudged in the other two.
An unknown value means keep (faces, plates and names blurred, the picture otherwise as shot); nothing in any mode but remove ever stops a tour.
"""
import os

MODES = ("blur", "remove", "keep")


def mode():
    v = os.environ.get("SHOP_TOUR_PEOPLE", "").strip().lower()
    return v if v in MODES else "keep"


def remove():
    return mode() == "remove"


def holds():
    """Only the paint-out mode holds stops back (and so only it runs the margin and has a review)."""
    return remove()


def sigma_frac():
    """The smudge's Gaussian sigma as a share of the picture's width (SHOP_TOUR_PEOPLE_SIGMA, default 0.02)."""
    try:
        return min(0.2, max(0.002, float(os.environ.get("SHOP_TOUR_PEOPLE_SIGMA", "") or 0.02)))
    except ValueError:
        return 0.02
