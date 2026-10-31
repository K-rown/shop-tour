"""Fill a hole in a picture with texture that belongs to the place: patch synthesis (coarse to fine, nearest patch found by brute force, on the GPU when there is one).

Used only by remove_others.py (SHOP_TOUR_REMOVE_OTHERS=1): where a person was painted out and no neighbouring stop saw what is behind him, the old fill was a smooth smudge
or a single copied piece, and a smooth smudge on a road is the one thing a depth model reads as a standing lump and a person's eye reads as a ghost.

    synth(img, hole, source=None) -> (new picture, a note)   (only the hole changes)

  img     uint8 BGR (h, w, 3)         hole    bool (h, w), the pixels to make new      source  bool (h, w) or None: where the texture may come from (None: everywhere that is not near the hole)
How: the picture is shrunk in steps; the smallest hole is filled by Telea inpainting (a rough colour for each place), then at every size each patch of the hole is replaced by the
patch of the source whose colours and grain it matches best (all of them are tried: a few hundred thousand; a patch from near the hole is preferred to one from far off, where the light
and the material are different), the picture is rebuilt from the votes of the overlapping patches (a sharp kernel: the grain stays), and this is repeated while going up in size.
The edge of the hole is not painted over: patches that touch it are matched on the real pixels beside it. At the end the colour of the real pixels at the edge is carried a little way
into the fill (a seam or a patch a shade too dark or too pink does not stand), the grain is not touched.
Nothing is random but the choice of which source patches to try when there are too many (a fixed seed): the same picture gives the same fill.
"""
import numpy as np
import cv2

PATCH = 7                       # patch side (pixels at every level)
MAX_SOURCES = 200_000           # source patches tried (a fixed random subset when there are more)
ITERS = (6, 5, 4, 2)            # iterations at the coarsest ... the finest level
MAX_LEVELS = 5
LOCAL = 2.0                     # weight of "copy from near the hole" against "the patch looks alike" (see _nearest)
CHUNK_BYTES = 700e6             # the biggest distance matrix held at once
EDGE_CARRY = 0.25               # how far into the fill (in hole sizes) the colour of the edge is carried


def _device():
    try:
        import torch
        return torch, ("cuda" if torch.cuda.is_available() else "cpu")
    except Exception:
        return None, None


def _levels(hole, patch):
    ys, xs = np.nonzero(hole)
    side = max(1, min(ys.max() - ys.min() + 1, xs.max() - xs.min() + 1))
    n = 0
    while side / (2 ** (n + 1)) >= 3 * patch and n < MAX_LEVELS - 1:
        n += 1
    return n


def _shrink(img, hole, source, k):
    """The picture, hole and source at 1/2**k size (the hole kept where any of it is, the source only where all of it is)."""
    if k == 0:
        return img, hole, source
    h, w = hole.shape
    s = 2 ** k
    sz = (max(8, w // s), max(8, h // s))
    im = cv2.resize(img, sz, interpolation=cv2.INTER_AREA)
    ho = cv2.resize(hole.astype(np.uint8), sz, interpolation=cv2.INTER_AREA) > 0
    so = cv2.resize(source.astype(np.uint8), sz, interpolation=cv2.INTER_AREA) >= 1
    return im, ho, so & ~ho


def _patches(t, img_f, cy, cx, r):
    """Patch vectors (n, 3*(2r+1)**2) of the float tensor image (3, h, w) at the centres (cy, cx), the picture reflected at its border."""
    k = 2 * r + 1
    dy, dx = t.meshgrid(t.arange(-r, r + 1, device=img_f.device), t.arange(-r, r + 1, device=img_f.device), indexing="ij")
    H, W = img_f.shape[1:]
    yy = (cy[:, None, None] + dy[None]).clamp(0, H - 1)
    xx = (cx[:, None, None] + dx[None]).clamp(0, W - 1)
    p = img_f[:, yy, xx]                                   # (3, n, k, k)
    return p.permute(1, 0, 2, 3).reshape(len(cy), 3 * k * k)


def _nearest(t, dev, tgt, src, tpos, spos, lam, chunk_bytes=CHUNK_BYTES):
    """Index of the closest source vector for every target vector: the squared difference of the patches (in 0..1 colour, 3*7*7 numbers) plus lam times the squared
    distance between the places (in units of the hole's size), so that what is copied comes from near the hole, where the light and the material are the same."""
    n_s = src.shape[0]
    ly, lx = (lam, lam) if not isinstance(lam, (tuple, list)) else lam      # (a number: the same weight up and across; two numbers: up/down, left/right - a street scene is made of horizontal strata)
    half = dev == "cuda"                                    # (half precision on the card: the closest of a few hundred thousand patches does not need more, and it is four times quicker)
    dt = t.float16 if half else t.float32
    s2 = (src * src).sum(1).to(dt)
    srch = src.to(dt)
    sp0, sp1 = spos[:, 0].to(dt), spos[:, 1].to(dt)
    per = max(64, int(chunk_bytes / (2 * n_s)))
    res = []
    for c0 in range(0, tgt.shape[0], per):
        a = tgt[c0:c0 + per].to(dt)
        d = s2[None, :] - 2.0 * (a @ srch.T)               # (the target's own square does not change which source is the closest)
        if ly > 0 or lx > 0:
            p = tpos[c0:c0 + per].to(dt)
            d = d + (ly * ((p[:, None, 0] - sp0[None, :]) ** 2) + lx * ((p[:, None, 1] - sp1[None, :]) ** 2))
        res.append(d.argmin(1))
    return t.cat(res) if res else t.zeros(0, dtype=t.long, device=dev)


def _blur_t(t, x, sigma):
    """Gaussian blur of a float tensor (3, h, w), reflected at the border."""
    rad = max(1, int(3 * sigma))
    ax = t.arange(-rad, rad + 1, device=x.device, dtype=x.dtype)
    g = t.exp(-(ax ** 2) / (2 * sigma * sigma)); g = g / g.sum()
    xp = t.nn.functional.pad(x[None], (rad, rad, rad, rad), mode="reflect")
    xp = t.nn.functional.conv2d(xp, g.reshape(1, 1, 1, -1).repeat(3, 1, 1, 1), groups=3)
    xp = t.nn.functional.conv2d(xp, g.reshape(1, 1, -1, 1).repeat(3, 1, 1, 1), groups=3)
    return xp[0]


FEATURE_BLUR = 1.5              # patches are MATCHED on the picture smoothed by this much (pixels) and the grain is taken from the source as it is: matched on grain, a smooth guess picks
                                # the smoothest sources, and averaged, noise turns into a smear (a road is nothing but noise)


def _level(t, dev, img, hole, source, iters, rng, stride, patch, lam, last=False):
    """_level_impl with the votes added up in a fixed order (the same picture gives the same fill)."""
    before = t.are_deterministic_algorithms_enabled()
    t.use_deterministic_algorithms(True, warn_only=True)
    try:
        return _level_impl(t, dev, img, hole, source, iters, rng, stride, patch, lam, last)
    finally:
        t.use_deterministic_algorithms(before)


def _level_impl(t, dev, img, hole, source, iters, rng, stride, patch, lam, last=False):
    """Improve the picture inside `hole` (uint8 BGR, already holding a first guess there) by `iters` rounds of nearest-patch voting; with last=True the final round copies the source's
    pixels as they are (a block of stride x stride from the middle of each chosen patch) instead of averaging the votes, so that the grain of the place is kept."""
    r = patch // 2
    H, W = hole.shape
    img_f = t.from_numpy(np.ascontiguousarray(img.transpose(2, 0, 1)).astype(np.float32) / 255.0).to(dev)
    raw0 = img_f.clone()
    blur0 = _blur_t(t, raw0, FEATURE_BLUR)
    # source patches: centres whose whole patch is source (the source mask eroded by the patch radius)
    ok = cv2.erode(source.astype(np.uint8), np.ones((patch, patch), np.uint8)) > 0
    sy, sx = np.nonzero(ok)
    if len(sy) < 50:
        return img
    if len(sy) > MAX_SOURCES:
        sel = rng.choice(len(sy), MAX_SOURCES, replace=False)
        sy, sx = sy[sel], sx[sel]
    sy_t, sx_t = t.from_numpy(sy).to(dev), t.from_numpy(sx).to(dev)
    src = _patches(t, blur0, sy_t, sx_t, r)                # what is matched
    srcr = _patches(t, raw0, sy_t, sx_t, r)                # what is copied
    ys0, xs0 = np.nonzero(hole)
    size = float(max(ys0.max() - ys0.min(), xs0.max() - xs0.min()) + 1)
    spos = t.stack([sy_t.float() / size, sx_t.float() / size], 1)
    # target centres: the hole, and the ring of known pixels whose patches overlap it (those are what ties the fill to its surroundings)
    near = cv2.dilate(hole.astype(np.uint8), np.ones((patch, patch), np.uint8)) > 0
    ty, tx = np.nonzero(near)
    if stride > 1:
        keep = ((ty % stride) == 0) & ((tx % stride) == 0)
        ty, tx = ty[keep], tx[keep]
    ty_t, tx_t = t.from_numpy(ty).to(dev), t.from_numpy(tx).to(dev)
    tpos = t.stack([ty_t.float() / size, tx_t.float() / size], 1)
    k = 2 * r + 1
    g = cv2.getGaussianKernel(k, 1.1)
    wk = t.from_numpy((g @ g.T).astype(np.float32)).to(dev).reshape(1, k, k)
    dy, dx = t.meshgrid(t.arange(-r, r + 1, device=dev), t.arange(-r, r + 1, device=dev), indexing="ij")
    hole_t = t.from_numpy(hole).to(dev)
    yy = (ty_t[:, None, None] + dy[None]); xx = (tx_t[:, None, None] + dx[None])
    inside = (yy >= 0) & (yy < H) & (xx >= 0) & (xx < W)
    flat = (yy.clamp(0, H - 1) * W + xx.clamp(0, W - 1)).reshape(-1)
    w = (wk * inside).reshape(-1)
    for it in range(iters):
        tgt = _patches(t, _blur_t(t, img_f, FEATURE_BLUR), ty_t, tx_t, r)
        nn = _nearest(t, dev, tgt, src, tpos, spos, lam)
        got = srcr[nn].reshape(len(ty), 3, k, k)           # the chosen source patches, grain and all
        if last and it == iters - 1:
            s = max(1, stride)
            new = img_f.clone()
            for oy in range(s):
                for ox in range(s):
                    py, px = ty_t + oy, tx_t + ox
                    okp = (py < H) & (px < W)
                    py, px = py.clamp(0, H - 1), px.clamp(0, W - 1)
                    take = okp & hole_t[py, px]
                    vals = got[:, :, r + oy, r + ox] if r + oy < k and r + ox < k else got[:, :, r, r]
                    idx = take.nonzero(as_tuple=True)[0]
                    new[:, py[idx], px[idx]] = vals[idx].T
            img_f = t.where(hole_t[None], new, img_f)
            break
        acc = t.zeros((3, H * W), device=dev); wsum = t.zeros(H * W, device=dev)
        for c in range(3):
            acc[c].index_add_(0, flat, ((got[:, c] * wk) * inside).reshape(-1))
        wsum.index_add_(0, flat, w)
        new = (acc / wsum.clamp_min(1e-6)).reshape(3, H, W)
        upd = hole_t & (wsum.reshape(H, W) > 1e-6)
        img_f = t.where(upd[None], new, img_f)
    out = (img_f.clamp(0, 1) * 255.0 + 0.5).byte().permute(1, 2, 0).cpu().numpy()
    return np.where(hole[..., None], out, img)


def carry_edge_colour(img, filled, hole):
    """The colour of the real pixels at the edge of the hole, carried a little way into the fill: where the fill's colour at its edge differs from the picture beside it (a patch from
    a sunlit place into a shadow), the difference is spread inwards, fading out over EDGE_CARRY hole sizes. The grain of the fill is not touched. Returns the new picture."""
    ho = hole.astype(np.uint8)
    if not ho.any():
        return filled
    d = cv2.distanceTransform(ho, cv2.DIST_L2, 3)
    ys, xs = np.nonzero(ho)
    size = float(max(ys.max() - ys.min(), xs.max() - xs.min()) + 1)
    band = (d > 0) & (d <= 4)
    out_ring = (cv2.dilate(ho, np.ones((9, 9), np.uint8)) > 0) & ~hole
    if band.sum() < 50 or out_ring.sum() < 50:
        return filled
    f = filled.astype(np.float32); o = img.astype(np.float32)
    s = max(1.5, 0.004 * size)
    hk = ho.astype(np.float32)                                   # the fill's own colour at the edge: blurred over the fill only (the real picture beside it is not mixed in)
    fb = cv2.GaussianBlur(f * hk[..., None], (0, 0), s) / np.maximum(cv2.GaussianBlur(hk, (0, 0), s), 1e-4)[..., None]
    # what the real picture holds just outside, spread to the band (a normalised blur of the ring)
    k = out_ring.astype(np.float32)
    num = cv2.GaussianBlur(o * k[..., None], (0, 0), 6.0); den = cv2.GaussianBlur(k, (0, 0), 6.0)
    ref = num / np.maximum(den[..., None], 1e-4)
    delta = np.clip(np.where(band[..., None], ref - fb, 0.0).astype(np.float32), -50, 50)
    kb = band.astype(np.float32)
    sg = max(4.0, 0.08 * size)
    nd = cv2.GaussianBlur(delta, (0, 0), sg); dd = cv2.GaussianBlur(kb, (0, 0), sg)
    field = nd / np.maximum(dd[..., None], 1e-4)
    fade = np.exp(-d / max(3.0, EDGE_CARRY * size))[..., None]
    out = f + field * fade * hole[..., None]
    return np.where(hole[..., None], np.clip(out, 0, 255).astype(np.uint8), filled)


def synth(img, hole, source=None, seed=0, patch=PATCH, local=LOCAL):
    """See the module's header. Returns (picture, note): the picture with the hole filled; note says how ('synth: 4 levels, 183k source px, cuda' or why nothing was done)."""
    hole = hole.astype(bool)
    if not hole.any():
        return img, "nothing to fill"
    t, dev = _device()
    if t is None:
        return img, "no torch"
    near = cv2.dilate(hole.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * patch + 1, 2 * patch + 1))) > 0
    if source is None:
        source = ~near
    else:
        source = source.astype(bool) & ~near
    if int(source.sum()) < 400:
        return img, "too little to copy from"
    rng = np.random.default_rng(seed)
    L = _levels(hole, patch)
    img0 = img.copy()
    cur = None
    note = ""
    for lv in range(L, -1, -1):
        im, ho, so = _shrink(img0, hole, source, lv)
        if lv == L:
            small = cv2.inpaint(im, cv2.dilate(ho.astype(np.uint8), np.ones((3, 3), np.uint8)), 3, cv2.INPAINT_TELEA)
            cur = np.where(ho[..., None], small, im)
        else:
            up = cv2.resize(cur, (im.shape[1], im.shape[0]), interpolation=cv2.INTER_CUBIC)
            cur = np.where(ho[..., None], up, im)
        stride = 2 if (lv == 0 and int(ho.sum()) > 60000) else 1
        try:
            cur = _level(t, dev, cur, ho, so, ITERS[min(len(ITERS) - 1, L - lv)], rng, stride, patch, local, last=(lv == 0))
        except Exception as e:                                # (a card with no room left: this level keeps the guess it had)
            if "out of memory" in str(e).lower() and dev == "cuda":
                t.cuda.empty_cache()
                note += f" level {lv} out of memory;"
            else:
                raise
    out = np.where(hole[..., None], cur, img)
    out = carry_edge_colour(img, out, hole)
    return out, f"synth: {L + 1} levels, {int(source.sum())} source px, {dev}{note}"
