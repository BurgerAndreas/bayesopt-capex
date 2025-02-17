
There are two ways to achieve the dashboard:
1. precompute the data and save it to a file, then read it in the dashboard
2. compute the data on the fly in the dashboard

### Precompute
six input variables -> six-dimensional grid 
resolution of e.g. 20 points per variable, 
total number of grid points is:
20^6 = 64,000,000 points.

If you store each scalar output in float16 (which uses 16 bits or 2 bytes per value), the total disk space required is:
64,000,000 points * 2 bytes/point = 128,000,000 bytes ~ 128 MB of disk space



