### COPYING

---

#### FlashTrie (MIT License)

The FlashTrie to marisa-trie — including GPU-accelerated beam search, CUDA kernels, and related additions — are released under the MIT License.

Copyright (c) 2026 Microsoft Corporation

Contributors: Dakshitha B Anandakumar (Microsoft), Anurag Mukkara (NVIDIA)

New files contributed by FlashTrie:
- `include/marisa/tbs_input.h`
- `lib/marisa/grimoire/cuda_allocator.h`
- `lib/marisa/grimoire/cuda_check.h`
- `lib/marisa/grimoire/trie/beam-search-state.h`
- `lib/marisa/grimoire/trie/beam-state.h`
- `lib/marisa/grimoire/trie/candidates.h`
- `lib/marisa/grimoire/trie/candidates-gpu.h`
- `lib/marisa/grimoire/trie/tbs-kernels.cu`
- `tools/beam-search-helper.h`
- `tools/marisa-beam-search.cc`
- `tools/bindings.cpp`

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

---

#### Original marisa-trie License

This repository is based on [marisa-trie](https://github.com/s-yata/marisa-trie) by Susumu Yata. The original marisa-trie code is licensed under BSD-2-Clause OR LGPL-2.1-or-later.

libmarisa and its command line tools are licensed under BSD-2-Clause OR LGPL-2.1-or-later.

#### The BSD 2-clause license

Copyright (c) 2010-2025, Susumu Yata
All rights reserved.

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

- Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
- Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#### The LGPL 2.1 or any later version

marisa-trie - A static and space-efficient trie data structure.
Copyright (C) 2010-2025  Susumu Yata

This library is free software; you can redistribute it and/or
modify it under the terms of the GNU Lesser General Public
License as published by the Free Software Foundation; either
version 2.1 of the License, or (at your option) any later version.

This library is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public
License along with this library; if not, write to the Free Software
Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301  USA
