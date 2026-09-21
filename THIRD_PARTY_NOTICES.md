# Third-party notices

Falcon-OCR model architecture and inference behavior are adapted from Technology
Innovation Institute's Falcon-OCR / Falcon-Perception project, licensed under
Apache License 2.0. Downloaded checkpoints and reference source remain under
their upstream licenses. Upstream revisions and hashes are recorded in
`reference/manifest.json`.

## Pillow / Python Imaging Library

The compatible bicubic resampler in `src/preprocess.rs` adapts the coefficient
calculation and fixed-point arithmetic from Pillow's `src/libImaging/Resample.c`.

Copyright © 1997-2011 by Secret Labs AB

Copyright © 1995-2011 by Fredrik Lundh and contributors

Copyright © 2010 by Jeffrey A. Clark and contributors

By obtaining, using, and/or copying this software and/or its associated
documentation, you agree that you have read, understood, and will comply with
the following terms and conditions:

Permission to use, copy, modify and distribute this software and its
documentation for any purpose and without fee is hereby granted, provided that
the above copyright notice appears in all copies, and that both that copyright
notice and this permission notice appear in supporting documentation, and that
the name of Secret Labs AB or the author not be used in advertising or publicity
pertaining to distribution of the software without specific, written prior
permission.

SECRET LABS AB AND THE AUTHOR DISCLAIMS ALL WARRANTIES WITH REGARD TO THIS
SOFTWARE, INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS. IN
NO EVENT SHALL SECRET LABS AB OR THE AUTHOR BE LIABLE FOR ANY SPECIAL, INDIRECT
OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES WHATSOEVER RESULTING FROM LOSS OF USE,
DATA OR PROFITS, WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS
ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS
SOFTWARE.

## libjpeg-turbo and Rust TurboJPEG bindings

This software is based in part on the work of the Independent JPEG Group.

JPEG decoding links libjpeg-turbo 3.1.0, built with SIMD from the sources vendored
by `turbojpeg-sys` 1.2.0 through the `turbojpeg` 1.5.1 Rust crate. The native
library's full license notices are included in
[libjpeg-turbo-LICENSE.md](licenses/libjpeg-turbo-LICENSE.md) and
[libjpeg-turbo-README.ijg](licenses/libjpeg-turbo-README.ijg). Both apply to binary
redistribution. The Rust bindings' MIT license is included in
[turbojpeg-MIT.txt](licenses/turbojpeg-MIT.txt).

The pinned GPU reference uses Pillow with libjpeg-turbo 3.1.4.1. Exact decoded RGB
and source-mode preprocessing match on the checked-in JPEG fixtures; this does
not assert equivalence for every possible JPEG bitstream.

Build helpers can download NASM 2.16.03 and CMake 3.31.10 into local tool caches.
Their source archives/distributions retain the upstream licenses. These tools
are build prerequisites and are not linked into the runner. Download versions
and archive SHA-256 values are fixed in the build helpers.
