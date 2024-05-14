# Simultaneous Localization and Mapping (SLAM)

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