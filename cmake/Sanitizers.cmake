# AddressSanitizer and UndefinedBehaviorSanitizer configuration

if(CE_SANITIZE)
  if(TARGET computeedge_compile_options)
    target_compile_options(computeedge_compile_options INTERFACE
      -fsanitize=address,undefined -fno-omit-frame-pointer -fno-sanitize-recover=undefined)
    target_link_options(computeedge_compile_options INTERFACE -fsanitize=address,undefined)
  endif()
endif()
