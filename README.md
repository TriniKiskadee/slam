# Simultaneous Localization and Mapping (SLAM)

## PC
- Ubuntu 24.04 (Kernel Ubuntu 6.8.0-31.31-generic 6.8.1)
- i5-4690K
- 32Gb RAM

## Technology Used
- Python 3.9
- CMake
- [Pangolin](https://github.com/stevenlovegrove/Pangolin)
- [G2OPY](https://github.com/uoip/g2opy)

## Major Errors and Fixes
### MESA-LOADER: failed to open iris
Error:
```console
    MESA-LOADER: failed to open iris: /usr/lib/dri/iris_dri.so: cannot open shared object file: No such file or directory (search paths /usr/lib/x86_64-linux-gnu/dri:\$${ORIGIN}/dri:/usr/lib/dri, suffix _dri)
    failed to load driver: iris
    MESA-LOADER: failed to open swrast: /usr/lib/dri/swrast_dri.so: cannot open shared object file: No such file or directory (search paths /usr/lib/x86_64-linux-gnu/dri:\$${ORIGIN}/dri:/usr/lib/dri, suffix _dri)
    X Error of failed request:  BadValue (integer parameter out of range for operation)
      Major opcode of failed request:  149 (GLX)
      Minor opcode of failed request:  3 (X_GLXCreateContext)
      Value in failed request:  0x0
      Serial number of failed request:  198
      Current serial number in output stream:  199
    
    Process finished with exit code 1
 ```

Solution: [StackOverflow](https://stackoverflow.com/questions/72110384/libgl-error-mesa-loader-failed-to-open-iris)
```console
conda install -c conda-forge libstdcxx-ng
```

### `GLIBCXX_3.4.32' not found
Error: 
```console
ImportError: /home/runie/.conda/envs/slam/bin/../lib/libstdc++.so.6: version `GLIBCXX_3.4.32' not found (required by /home/runie/.conda/envs/slam/lib/python3.9/site-packages/pypangolin.cpython-39-x86_64-linux-gnu.so)
```

Solution: [StackOverflow](https://askubuntu.com/questions/1418016/glibcxx-3-4-30-not-found-in-conda-environment)
```console
ln -sf /usr/lib/x86_64-linux-gnu/libstdc++.so.6 /home/anavani/anaconda3/envs/dmcgb/bin/../lib/libstdc++.so.6
ln -sf /usr/lib/x86_64-linux-gnu/libgcc_s.so.1 /home/runie/.conda/envs/slam/bin/../lib/libgcc_s.so.1
```