#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <chrono>
#include <algorithm>
#include "rkllm.h"
static std::vector<float> hidden; static int g_tokens=0,g_dim=0;
static int cb(RKLLMResult* r,void*,LLMCallState s){if(s==RKLLM_RUN_NORMAL&&r&&r->last_hidden_layer.hidden_states){g_tokens=r->last_hidden_layer.num_tokens;g_dim=r->last_hidden_layer.embd_size;hidden.assign(r->last_hidden_layer.hidden_states,r->last_hidden_layer.hidden_states+(size_t)g_tokens*g_dim);}return 0;}
static float lin(const float*w,const float*h,const float*b){float x=*b;for(int i=0;i<1024;i++)x+=w[i]*h[i];return x;}
int main(int argc,char**argv){if(argc<5){fprintf(stderr,"usage: %s model head manifest ids_dir\n",argv[0]);return 2;}RKLLMParam p=rkllm_createDefaultParam();p.model_path=argv[1];p.max_context_len=8192;p.max_new_tokens=1;p.skip_special_token=false;RKLLMCallback c{};c.result_callback=cb;LLMHandle h=nullptr;int ret=rkllm_init(&h,&p,&c);printf("init_ret=%d\n",ret);if(ret)return 3;
std::ifstream fh(argv[2],std::ios::binary);std::vector<float>qW(256*1024),qb(256),kW(256*1024),kb(256);float temp;fh.read((char*)qW.data(),qW.size()*4);fh.read((char*)qb.data(),qb.size()*4);fh.read((char*)kW.data(),kW.size()*4);fh.read((char*)kb.data(),kb.size()*4);fh.read((char*)&temp,4);
std::ifstream mf(argv[3]);std::string line,dir=argv[4];while(std::getline(mf,line)){if(line.empty())continue;std::stringstream ss(line);std::string tag;int decide;ss>>tag>>decide;std::vector<int>opts;int oi;while(ss>>oi)opts.push_back(oi);std::ifstream fi(dir+"/"+tag+".txt");std::vector<int32_t>ids;int x;while(fi>>x)ids.push_back(x);RKLLMInput in{};in.role="user";in.enable_thinking=false;in.input_type=RKLLM_INPUT_TOKEN;in.token_input.input_ids=ids.data();in.token_input.n_tokens=ids.size();RKLLMInferParam ip{};ip.mode=RKLLM_INFER_GET_LAST_HIDDEN_LAYER;ip.keep_history=0;ip.max_new_tokens=1;rkllm_clear_kv_cache(h,0,nullptr,nullptr);auto t0=std::chrono::steady_clock::now();g_tokens=g_dim=0;ret=rkllm_run(h,&in,&ip,nullptr);double ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-t0).count();std::vector<float> probs;int chosen=-1;if(ret==0&&g_tokens==(int)ids.size()&&g_dim==1024){const float*hd=&hidden[(size_t)decide*1024];std::vector<float>q(256);for(int j=0;j<256;j++)q[j]=lin(&qW[(size_t)j*1024],hd,&qb[j]);std::vector<float>z(opts.size());for(size_t o=0;o<opts.size();o++){const float*ho=&hidden[(size_t)opts[o]*1024];float v=0;for(int j=0;j<256;j++)v+=lin(&kW[(size_t)j*1024],ho,&kb[j])*q[j];z[o]=v/std::sqrt(256.f)/temp;}float mx=*std::max_element(z.begin(),z.end()),sum=0;for(float&v:z){v=std::exp(v-mx);sum+=v;}for(float&v:z){v/=sum;probs.push_back(v);}chosen=std::max_element(probs.begin(),probs.end())-probs.begin();}printf("row=%s ret=%d input=%zu hidden=%d dim=%d ms=%.3f chosen_index=%d probs=",tag.c_str(),ret,ids.size(),g_tokens,g_dim,ms,chosen);for(float v:probs)printf("%.9g,",v);printf("\n");fflush(stdout);}rkllm_destroy(h);return 0;}
