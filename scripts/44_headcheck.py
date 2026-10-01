"""Verify an RKNN file is not truncated, using the size the header itself declares.

The s90 export that failed silently on a full disk had export_rknn() return 0 and the tool
log report 472469504 bytes while the header at offset 16 declared 658647040 -- the device
only found out at parseRKNN time. So for every produced .rknn: file size must be >= the
declared data size, and the gap (header overhead) must be small.
"""
import hashlib
import struct
import sys


def md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def check(path):
    size = 0
    with open(path, "rb") as f:
        size = f.seek(0, 2)
        f.seek(0)
        magic = f.read(4)
        f.seek(16)
        off16 = struct.unpack("<Q", f.read(8))[0]
        f.seek(0)
        head16 = f.read(16).hex()
    delta = size - off16
    verdict = "OK" if (off16 > 0 and 0 <= delta <= 262144) else "TRUNCATED_OR_BAD"
    print("%-70s size=%-12d off16=%-12d delta=%-8d magic=%-8r head16=%s %s md5=%s"
          % (path.split("/")[-1], size, off16, delta, magic, head16, verdict, md5(path)),
          flush=True)
    return verdict, size, off16, delta


def main():
    bad = 0
    for p in sys.argv[1:]:
        try:
            v, _, _, _ = check(p)
            if v != "OK":
                bad += 1
        except Exception as e:
            print("%s CHECK_FAILED %s: %s" % (p, type(e).__name__, e), flush=True)
            bad += 1
    print("### HEADCHECK %s (%d files, %d bad)" % ("PASS" if bad == 0 else "FAIL",
                                                   len(sys.argv) - 1, bad), flush=True)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
