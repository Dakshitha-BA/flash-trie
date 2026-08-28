SUBDIRS = grimoire

MY_INCLUDE = -I$(top_srcdir)/include -I$(top_srcdir)/lib @CUDA_INCLUDES@

AM_CXXFLAGS = -Wall -Weffc++ -Wextra -Wconversion $(MY_INCLUDE)

lib_LTLIBRARIES = libmarisa.la

libmarisa_la_LDFLAGS = -no-undefined @CUDA_LDFLAGS@ @CUDA_LIBS@

libmarisa_la_LIBADD = \
  grimoire/io/libio.la \
  grimoire/trie/libtrie.la \
  grimoire/vector/libvector.la

libmarisa_la_SOURCES = \
  keyset.cc \
  agent.cc \
  trie.cc
