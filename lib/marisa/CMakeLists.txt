add_subdirectory(grimoire/io)
add_subdirectory(grimoire/trie)
add_subdirectory(grimoire/vector)

set(LIB_MARISA marisa)
set(LIB_MARISA_SRCS keyset.cc agent.cc trie.cc)
set(LIB_MARISA_DEPENDS io trie vector)

add_library(${LIB_MARISA} STATIC ${LIB_MARISA_SRCS})
target_include_directories(${LIB_MARISA} PUBLIC /usr/local/cuda/include ${CMAKE_SOURCE_DIR}/include ${CMAKE_SOURCE_DIR}/lib)
target_link_libraries(${LIB_MARISA} PRIVATE ${LIB_MARISA_DEPENDS})
