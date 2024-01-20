#include<bits/stdc++.h>
using namespace std;

void back(int w, int d,string pattern, vector<string>&ans,string temp,int i)
{
    if(i==6)
    {
        if(w==0)
        {
            ans.push_back(temp);
            return;
        }
        else
        {
            return;
        }
       
    }

    
    if(pattern[i]=='?')
    {
         for(int j=0;j<=d;j++)
       {
           if(j>w)
           {
            break;
           }
           else
           {
             char a=to_char(j);
             temp.push_back(a);
             back(w-j,d,pattern,ans,temp,i+1);
           }
       }
    }
    else
    {
        int a=to_int(pattern[i]);
        temp.push_back(pattern[i]);
        back(w-a,d,pattern,ans,temp,i+1);
    }
    
}
vector<string> func(int w, int d,string pattern)
{
    vector<string>ans;
    string temp;
    back(w,d,pattern,ans,temp,0);
    return ans;
}
int main()
{
    ios_base::sync_with_stdio(false);cin.tie(NULL);cout.tie(NULL);
    cout<<fixed<<setprecision(25);
    cerr<<fixed<<setprecision(10);
    auto start= std::chrono::high_resolution_clock::now();
    
    long long int t;
    cin>>t;
    
    while(t>0)
    {
      int w;
      int d;
      string inp;
      cin>>w>>d>>inp;
      

      cout<<func(w,d,inp)<<endl;
       
      
      
       
       t--;
    }
    
    
}