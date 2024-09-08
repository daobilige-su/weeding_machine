#include <iostream>
#include <cmath>

const double pi = 3.141592;

//for u_idx in range(x_range.shape[0]):
//    u = x_range[u_idx]
//
//    for theta_idx in range(theta_range.shape[0]):
//        theta = theta_range[theta_idx]
//
//        # x = -np.tan(np.deg2rad(theta)) * y_range + (u + np.tan(np.deg2rad(theta)) * (img_size[0]-1))
//        x = -np.tan(np.deg2rad(theta)) * y_range + (u + np.tan(np.deg2rad(theta)) * 0)
//        y = y_range.copy()
//
//        x_valid = x[(0 <= x) & (x < img_size[1])]
//        y_valid = y[(0 <= x) & (x < img_size[1])]
//
//        x_valid = x_valid[(0 <= y_valid) & (y_valid < img_size[0])]
//        y_valid = y_valid[(0 <= y_valid) & (y_valid < img_size[0])]
//
//        his[u_idx, theta_idx] = np.sum(plant_seg_guas[(img_size[0]-1)-y_valid.astype(int), x_valid.astype(int)])
//        ll_heur = his[u_idx, theta_idx]
//
//        for shift_idx in range(shift_range.shape[0]):
//            shift = shift_range[shift_idx]
//
//            rl_u = u + self.wrap_img_size[1]/2.0 + shift  # right line bottom u
//
//            inter_u = u + inter_h * np.tan(-np.deg2rad(theta))  # u of the top intersection point of left and right line
//
//            rl_theta = np.rad2deg(np.arctan2(rl_u - inter_u, inter_h))
//
//            x = -np.tan(np.deg2rad(rl_theta)) * y_range + (rl_u + np.tan(np.deg2rad(rl_theta)) * 0)
//            y = y_range.copy()
//
//            x_valid = x[(0 <= x) & (x < img_size[1])]
//            y_valid = y[(0 <= x) & (x < img_size[1])]
//
//            x_valid = x_valid[(0 <= y_valid) & (y_valid < img_size[0])]
//            y_valid = y_valid[(0 <= y_valid) & (y_valid < img_size[0])]
//
//            rl_heur = np.sum(plant_seg_guas[(img_size[0] - 1) - y_valid.astype(int), x_valid.astype(int)])
//
//            his_3d[u_idx, theta_idx, shift_idx] = ll_heur + rl_heur

extern "C" {
    int GrayScaleHoughLine(int img_size_u, int img_size_v, int x_range_num, int y_range_num, double theta_range[], int theta_range_num,
                           double shift_range[], int shift_range_num, double inter_h, double plant_seg_guas[], double *his_np) // plant_seg_guas: row first, v*u
    {
        double u, theta, shift, v, x, y, inter_u, rl_u, rl_theta;
        int x_int, y_int;
        double ll_heur, rl_heur;

        //double* his = new double[x_range_num*theta_range_num];
        for(int n=0; n<x_range_num*theta_range_num*shift_range_num; n++){
            his_np[n] = 0.0;
            // std::cout << his_np[n] << ",";
        }
        //std::cout << x_range_num*theta_range_num << std::endl;

        for(int u_idx=0; u_idx<x_range_num; u_idx++){
            u = u_idx;
//            std::cout << std::endl;
//            std::cout << "u" << u << std::endl;

            for(int theta_idx=0; theta_idx<theta_range_num; theta_idx++){
                theta = theta_range[theta_idx];

//                std::cout << theta_idx << ",";
//                std::cout << theta << ",";

                ll_heur = 0;

                for(int v_idx=0; v_idx<y_range_num; v_idx++){
                    v = v_idx;
//                    std::cout << "v" << v << std::endl;

                    y = v;
                    x = (-1.0)*tan( theta*(pi/180.0) )*y + u;

                    y_int = int(y);
                    x_int = int(x);

                    if((x_int>=0) && (x_int<img_size_u) && (y_int>=0) && (y_int<img_size_v)){
//                        his_np[u_idx*theta_range_num + theta_idx] = his_np[u_idx*theta_range_num + theta_idx] + plant_seg_guas[((y_range_num-1)-y_int)*x_range_num + x_int];
                        ll_heur = ll_heur + plant_seg_guas[((img_size_v-1)-y_int)*img_size_u + x_int];
                    }

                }

                for(int shift_idx=0; shift_idx<shift_range_num; shift_idx++){
                    shift = shift_range[shift_idx];

                    rl_u = u + img_size_u/2.0 + shift;

                    inter_u = u + inter_h * tan(-1.0*theta*(pi/180.0));

                    rl_theta = (180.0/pi)*(atan2(rl_u - inter_u, inter_h));

                    rl_heur = 0;

                    for(int v_idx=0; v_idx<y_range_num; v_idx++){
                        v = v_idx;

                        x = (-1.0)*tan( rl_theta*(pi/180.0) )*v + rl_u;
                        y = v;

                        y_int = int(y);
                        x_int = int(x);

                        if((x_int>=0) && (x_int<img_size_u) && (y_int>=0) && (y_int<img_size_v)){
                            rl_heur = rl_heur + plant_seg_guas[((img_size_v-1)-y_int)*img_size_u + x_int];
                        }

                        his_np[u_idx*theta_range_num*shift_range_num + theta_idx*shift_range_num + shift_idx] = ll_heur + rl_heur;
                    }

                }
            }
        }

//        for(int i=0; i<x_range_num; i++){
//            for(int j=0; j<theta_range_num; j++){
//                his_np[i*theta_range_num+j] = his[i*theta_range_num+j];
//            }
//        }

//        std::cout << his_np[0] << ", " << his_np[1] << ", " << his_np[2] << std::endl;
//        std::cout << his_np[theta_range_num+0] << ", " << his_np[theta_range_num+1] << ", " << his_np[theta_range_num+2] << std::endl;
//        std::cout << his_np[2*theta_range_num+0] << ", " << his_np[2*theta_range_num+1] << ", " << his_np[2*theta_range_num+2] << std::endl;

        return 0;
    }
}