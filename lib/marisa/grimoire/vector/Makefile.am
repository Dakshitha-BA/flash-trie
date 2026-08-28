MY_INCLUDE = -I$(top_srcdir)/include -I$(top_srcdir)/lib @CUDA_INCLUDES@

AM_CXXFLAGS = -Wall -Weffc++ -Wextra -Wconversion $(MY_INCLUDE)

noinst_LTLIBRARIES = libvector.la

libvector_la_LDFLAGS = -no-undefined @CUDA_LDFLAGS@ @CUDA_LIBS@

libvector_la_SOURCES = \
  bit-vector.cc

noinst_HEADERS = \
  pop-count.h \
  rank-index.h \
  vector.h \
  flat-vector.h \
  bit-vector.h
