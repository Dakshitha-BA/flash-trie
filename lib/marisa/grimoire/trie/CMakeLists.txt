set(LIB_TRIE trie)
set(LIB_TRIE_SRCS tail.cc louds-trie.cc)

add_library(${LIB_TRIE} STATIC ${LIB_TRIE_SRCS})
target_include_directories(${LIB_TRIE} PUBLIC /usr/local/cuda/include ${CMAKE_SOURCE_DIR}/include ${CMAKE_SOURCE_DIR}/lib)

set_source_files_properties(louds-trie.cc PROPERTIES LANGUAGE CUDA)
set_source_files_properties(tail.cc PROPERTIES LANGUAGE CUDA)
