# 混合并行模式的分布式机器学习加速方案

**环境配置**
预先安装 PyTorch （推荐1.10版本以上）
指定GLOO通信的网卡： export GLOO_SOCKET_IFNAME=eth3
指定Woker ID： export WID= 0|1|2|3

**运行命令**
python Flex_demo.py --wid $WID --wn 4 --bs 4 --subbs 4 --prt 23112 --ip 172.16.0.63

python worker-allreduce-baseline.py --wid $WID --wn 4 --bs 4 --nproc 1 --prt 23112  --ip 172.16.0.63

其中
--wid： Worker ID， 从0 开始
--wn: Worker 总数
--bs：每个worker一次迭代的Batch大小
--subbs: 在Flex方案下，每词迭代是拆分成多个自迭代的，subbs为每个子迭代的batch大小
--ip： 通信的IP，对应上文设定的eth3的IP
--prt： 通信的端口
--nproc: 使用的进程数目（deprecated）

**Nvidia-SMI 采样记录GPU利用率的命令**
nvidia-smi pmon -i 0 -c 100 -f flex-prof-blog.csv
nvidia-smi pmon -i 0 -c 100 -f base32-prof-blog.csv
详情参考 nvidia-smi --help-query–gpu

**相关文件**
原始测试数据以及相关图表，参见 [**MAXPP.xls**](https://github.com/Steamgjk/Hove/blob/master/MAXPP.xlsx "**MAXPP.xls**")
常用命令 参见 [**常用命令**](https://github.com/Steamgjk/Hove/blob/master/%E5%B8%B8%E7%94%A8%E5%91%BD%E4%BB%A4 "**常用命令**") 文件


##参考文献
 Jinkun Geng, Dan Li, Shuai Wang. [ElasticPipe: An Efficient and Dynamic Model-Parallel Solution to DNN Training.](https://cloud.tsinghua.edu.cn/f/4944219b08644faabe3c/?dl=1 "ElasticPipe: An Efficient and Dynamic Model-Parallel Solution to DNN Training. 10th workshop on Scientific Cloud Computing")  10th workshop on Scientific Cloud Computing ([HPDC](http://www.hpdc.org/2019/ "HPDC") Workshop on[ ScienceCloud'19](https://sites.google.com/view/sciencecloud2019 " ScienceCloud'19")) 
 
 Jinkun Geng, Dan Li, Shuai Wang. [Horizontal or Vertical? A Hybrid Approach to Large-Scale Distributed Machine Learning](https://cloud.tsinghua.edu.cn/f/f9ee738580b54bc1bb9e/?dl=1 "Horizontal or Vertical? A Hybrid Approach to Large-Scale Distributed Machine Learning"). 1st Workshop on Converged Computing Infrastructure ([HPDC](http://www.hpdc.org/2019/ "HPDC") Workshop on [CCIW'19](https://cciw.github.io/ "CCIW'19")) 
 
 更多相关论文可以通过访问 [个人主页](https://www.gengjinkun.com/ "个人主页") 获取
 https://www.gengjinkun.com/