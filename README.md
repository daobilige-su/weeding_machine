# weeding_machine

##setup conda env:

conda create -n weeding_machine python=3.8

conda activate weeding_machine

pip install -r <package_path>/yolov5/requirements.txt

pip install rospkg catkin_tools 

(these tools are not on path of ros packages on /opt/ros..., but on python path on
/usr/lib/python3/dist-packages/rospkg or 
anaconda3/env/env_name/lib/python3.8/site-packages/rospkg etc., so as you installed a
new python, these packages need to be re-installed again.
PS. to check a package's path: import rospkg & print(rospkg.__file__))

cd anaconda3/env/env_name/lib/

mv libffi.7.so libffi.7.so_bak
mv libffi.so.7 libffi.so.7_bak
ln -s /lib/x86_64-linux-gnu/libffi.so.7.1.0 libffi.7.so
ln -s /lib/x86_64-linux-gnu/libffi.so.7.1.0 libffi.so.7

(back up libffi.7.so & libffi.so.7, and point them to 
/lib/x86_64-linux-gnu/libffi.so.7.1.0
they are pointing to libffi.so.8.X.X contained in the new conda env which is wrong,
package like cv_bridge will get error like: 
libp11-kit.so.0: undefined symbol: ffi_type_pointer)

## Arduino to test serial comm:






