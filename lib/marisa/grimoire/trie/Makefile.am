MY_INCLUDE = -I$(top_srcdir)/include -I$(top_srcdir)/lib @CUDA_INCLUDES@

AM_CXXFLAGS = -Wall -Weffc++ -Wextra -Wconversion $(MY_INCLUDE)

noinst_LTLIBRARIES = libtrie.la

libtrie_la_LDFLAGS = -no-undefined @CUDA_LDFLAGS@ @CUDA_LIBS@

libtrie_la_SOURCES = \
  tail.cc \
  louds-trie.cc

noinst_HEADERS = \
  config.h \
  header.h \
  key.h \
  range.h \
  entry.h \
  tail.h \
  cache.h \
  history.h \
  state.h \
  louds-trie.h \
  tbs-kernels.cu
