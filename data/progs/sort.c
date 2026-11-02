void bubble(int *a, int n){for(int i=0;i<n;i++)for(int j=0;j+1<n-i;j++)if(a[j]>a[j+1]){int t=a[j];a[j]=a[j+1];a[j+1]=t;}}
int partition(int *a,int lo,int hi){int p=a[hi],i=lo-1;for(int j=lo;j<hi;j++)if(a[j]<=p){i++;int t=a[i];a[i]=a[j];a[j]=t;}int t=a[i+1];a[i+1]=a[hi];a[hi]=t;return i+1;}
void quick(int *a,int lo,int hi){if(lo<hi){int p=partition(a,lo,hi);quick(a,lo,p-1);quick(a,p+1,hi);}}
int sorted(int *a,int n){for(int i=1;i<n;i++)if(a[i-1]>a[i])return 0;return 1;}
