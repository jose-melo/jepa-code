unsigned djb2(const char *s){unsigned h=5381;int c;while((c=*s++))h=((h<<5)+h)+c;return h;}
unsigned fnv(const unsigned char *s,int n){unsigned h=2166136261u;for(int i=0;i<n;i++){h^=s[i];h*=16777619u;}return h;}
int count_char(const char *s,char c){int k=0;for(;*s;s++)if(*s==c)k++;return k;}
int strlen2(const char *s){int n=0;while(s[n])n++;return n;}
