#include <stdio.h>
#include <stdlib.h>
#include <malloc.h>

int val[100005];

int com_baiyi_findscore(const int* scorelist, int _sizeofscorelist){
    int i=0;
    for(i=0; i<_sizeofscorelist; i++){
        val[scorelist[i]]++;
    }
    int ans = 0;
    for(i=0;i<=1e5;i++){
        if(val[i] > _sizeofscorelist/2){
            ans = i;
        }
    }
    return ans;
}

// int main(){
//     int a[4] = {1, 1, 2, 2};
//     printf("%d", com_baiyi_findscore(a, 4));
//     return 0;
// }