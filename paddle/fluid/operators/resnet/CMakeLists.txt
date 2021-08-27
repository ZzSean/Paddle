include(operators)
if(WITH_UNITY_BUILD)
    # Load Unity Build rules for operators in paddle/fluid/operators/resnet.
    include(unity_build_rule.cmake)
endif()
register_operators(EXCLUDES
    cudnn_resnet_unit)

if (WITH_GPU OR WITH_ROCM)
    # cudnn_resnet_unit needs cudnn 8.0 above
    if (NOT ${CUDNN_VERSION} VERSION_LESS 8000)
        op_library(cudnn_resnet_unit)
        file(APPEND ${pybind_file} "USE_CUDA_ONLY_OP(cudnn_resnet_unit);\n")
    endif()
endif()
