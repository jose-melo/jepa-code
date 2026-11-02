void matmul(const double *A,const double *B,double *C,int n){for(int i=0;i<n;i++)for(int j=0;j<n;j++){double s=0;for(int k=0;k<n;k++)s+=A[i*n+k]*B[k*n+j];C[i*n+j]=s;}}
void transpose(double *A,int n){for(int i=0;i<n;i++)for(int j=i+1;j<n;j++){double t=A[i*n+j];A[i*n+j]=A[j*n+i];A[j*n+i]=t;}}
double trace(const double *A,int n){double s=0;for(int i=0;i<n;i++)s+=A[i*n+i];return s;}
