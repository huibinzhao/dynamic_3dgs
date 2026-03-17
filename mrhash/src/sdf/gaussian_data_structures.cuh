#pragma once
#include "voxel_data_structures.cuh"

#include <gaussian.cuh>
#include <loss_utils.cuh>
#include <quad_tree.cuh>
#include <render_utils.cuh>

#include <opencv2/core.hpp>

namespace cupanutils {
  namespace cugeoutils {

    template <typename T, typename Enable = void>
    class GaussianContainer {
      GaussianContainer() {
        std::cerr << "GaussianContainer is not implemented for this type" << std::endl;
      }
    };

    template <typename T>
    class GaussianContainer<T, std::enable_if_t<is_voxel_derived<T>::value>> {
      // gs shit
    public:
      GaussianContainer(std::string gs_optimization_param_path);
      ~GaussianContainer();
      void runGS(Camera& camera,
                 VoxelContainer<T>& container,
                 cupanutils::cugeoutils::CUDAMatrixuc3& rgb_img,
                 cupanutils::cugeoutils::CUDAMatrixf& depth_img,
                 cupanutils::cugeoutils::CUDAMatrixb* dynamic_mask = nullptr);
      void setupGSCamera(Camera& camera);
      void extractNodesQTree(Camera& camera,
                             VoxelContainer<T>& container,
                             cupanutils::cugeoutils::CUDAMatrixuc3& rgb_img,
                             cupanutils::cugeoutils::CUDAMatrixf& depth_img,
                             cupanutils::cugeoutils::CUDAMatrixb* dynamic_mask = nullptr);
      void optimizeGS();
      void optimizeGSFinal();
      __host__ void checkNodes(const Camera& camera,
                               const VoxelContainer<T>& container,
                               const CUDAMatrixuc3& rgb_img,
                               const CUDAMatrixf& depth_img,
                               const CUDAMatrixb* dynamic_mask = nullptr);
      std::vector<uint8_t> color_data_;
      std::vector<gs::Camera> gs_cam_list_;
      std::vector<torch::Tensor> gt_img_list_;
      gs::Camera cur_gs_cam_;
      gs::GaussianModel gs_model_;
      torch::Tensor cur_gt_img_;
      gs::CUDANode* d_qtree_nodes_   = nullptr;
      size_t num_qtree_nodes_        = 0;
      size_t num_valid_qtree_nodes_  = 0;
      size_t* d_num_qtree_nodes_     = nullptr;
      uint* d_num_valid_qtree_nodes_ = nullptr;
      gs::Point* d_positions_        = nullptr;
      gs::Color* d_colors_           = nullptr;
      float* d_scales_               = nullptr;
      torch::Tensor cur_mask_tensor_;
      std::vector<torch::Tensor> mask_list_;
      cv::Mat last_rendered_img_;
      bool has_rendered_img_ = false;
    };

    template class GaussianContainer<Voxel>;
    using GeometricGaussianContainer = GaussianContainer<Voxel>;
  } // namespace cugeoutils
} // namespace cupanutils
