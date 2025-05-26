#include<iostream>
using namespace std;
int n;
int e[1000], l[1000], d[1000];
int cost[1000][1000];
int path[1000];
bool Visited[1000] = {};
int cur_time = 0;
int Min_Time = __INT_MAX__;
int solution[1000];
int min_cost = __INT_MAX__;
int min_d = __INT_MAX__;
bool CheckIfValid(int cur_node, int cur_time, int last_node){
    if (!Visited[cur_node] && cur_time + cost[last_node][cur_node] + d[cur_node] <= l[cur_node]){
        return true;
    }
    return false;
}
void backTrack(int cur_step){
    int last_node = path[cur_step-1];

    for(int i = 1; i <=n ; i++){
        if (CheckIfValid(i,cur_time,last_node)){
            int old_curtime = cur_time;
            if (cur_time + cost[last_node][i] + d[i] < e[i]){
                    cur_time = e[i];   
            }
            else{
            cur_time += cost[last_node][i] + d[i];
            }

            Visited[i] = true;
            path[cur_step] = i;
            if(cur_step == n){
                if (cur_time + cost[i][0] < Min_Time){
                    Min_Time = cur_time + cost[i][0];
                    for(int j = 1; j<=n ; j++){
                        solution[j] = path[j];
                    }
                }
            }else{
                if(cur_time + (n - cur_step)*(min_d + min_cost) + min_cost < Min_Time){
                    backTrack(cur_step + 1);
                }
            }
            cur_time  = old_curtime;
            Visited[i] = false;
            path[cur_step] = 0;
        }
    }
}
int main(){
    cin>>n;
    for (int i=1; i<=n; i++){
       cin>>e[i]>>l[i]>>d[i];
       min_d = min(d[i], min_d);
    }
    for (int i=0; i<=n ; i++){
        for (int j=0; j<=n ; j++){
            cin>>cost[i][j];
            min_cost = min(min_cost,cost[i][j]);
        }
    }
    backTrack(1);
    cout<<n<<endl;
    for(int i = 1; i<=n ; i++ ){
        cout<<solution[i]<<" ";
    }
    return 0;

}