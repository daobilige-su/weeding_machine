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

extern "C" {
    int GrayScaleHoughLine(int x_range_num, int y_range_num, double theta_range[], int theta_range_num, double plant_seg_guas[], double *his_np) // plant_seg_guas: row first, v*u
    {
        double u, theta, v, x, y;
        int x_int, y_int;

        //double* his = new double[x_range_num*theta_range_num];
        for(int n=0; n<x_range_num*theta_range_num; n++){
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

                for(int v_idx=0; v_idx<y_range_num; v_idx++){
                    v = v_idx;
//                    std::cout << "v" << v << std::endl;

                    y = v;
                    x = (-1.0)*tan( theta*(pi/180.0) )*y + u;

                    y_int = int(y);
                    x_int = int(x);

                    if((x_int>=0) && (x_int<x_range_num) && (y_int>=0) && (y_int<y_range_num)){
                        his_np[u_idx*theta_range_num + theta_idx] = his_np[u_idx*theta_range_num + theta_idx] + plant_seg_guas[((y_range_num-1)-y_int)*x_range_num + x_int];
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