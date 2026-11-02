static int square(int x){return x*x;}
static int cube(int x){return x*square(x);}
int poly(int x){int r=0;for(int i=0;i<4;i++)r+=cube(x)+square(i)*x;return r;}
long fib(int n){long a=0,b=1;for(int i=0;i<n;i++){long t=a+b;a=b;b=t;}return a;}
int gcd(int a,int b){while(b){int t=a%b;a=b;b=t;}return a;}
int isprime(int n){if(n<2)return 0;for(int d=2;d*d<=n;d++)if(n%d==0)return 0;return 1;}
int sumprimes(int n){int s=0;for(int i=0;i<n;i++)if(isprime(i))s+=i;return s;}
