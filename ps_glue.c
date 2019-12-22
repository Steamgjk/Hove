//gcc -shared -o ps_glue.so -fPIC ps_glue.c -lpthread
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <pthread.h>
#include <unistd.h>
#include <sys/shm.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <signal.h>
#include <unistd.h>
#include <stdlib.h>
#include <assert.h>
#include <time.h>
#include <sys/time.h>

#define MEM_KEY (9062019)
#define MEM_CNT (256)
#define MEM_UNIT_SIZE (1024*1024*100)
#define BUFFER_SIZE (MEM_CNT*MEM_UNIT_SIZE) //500MB
#define LABEL_LEN (32)
#define CAP (16)

char* ps_ips[CAP] = {"127.0.0.1", "127.0.0.1", "127.0.0.1", "127.0.0.1"};
char* worker_ips[CAP] = {"127.0.0.1", "127.0.0.1", "127.0.0.1", "127.0.0.1"};
int to_worker_ports[CAP] = {5510, 5511, 5512, 5513};
int to_ps_ports[CAP] = {6610, 6611, 6612, 6613};

int ps_id = 0;
int ps_num = 1;
int worker_num = 4;
char* send_mem_ptr[CAP][MEM_CNT];
char* recv_mem_ptr[CAP][MEM_CNT];
int send_mem_len[CAP][MEM_CNT];
int recv_mem_len[CAP][MEM_CNT];
int send_mem_enque_idx[CAP];
int send_mem_deque_idx[CAP];
int recv_mem_enque_idx[CAP];
int recv_mem_deque_idx[CAP];
char* header_cache[CAP];

int init_info(int p_id, int p_num, int w_num);
int init_mem();
int launch_send_thread(int send_channel_id);
int launch_recv_thread(int recv_channel_id);
void* recv_td(void* args);
void* send_td(void* args);
int inquire_recv_mem_len(int worker_id);
int dequeue_recv_mem(int worker_id, char* data_out, int data_len);
int enque_send_mem(int worker_id, char* data_in, int in_len);



void recv_fixed_len(int connfd, char* sta_buf, int len);
int wait4connection(char*local_ip, int local_port);
int go4connection(char* remote_ip, int remote_port);

struct Header
{
	size_t data_len;
};
const int head_sz = sizeof(struct Header);

int init_info(int p_id, int p_num, int w_num)
{
	ps_id = p_id;
	ps_num = p_num;
	worker_num = w_num;
	return 0;
}
int init_mem()
{

	for (int i = 0; i < worker_num; i++)
	{
		send_mem_enque_idx[i] = 0;
		send_mem_deque_idx[i] = 0;
		recv_mem_enque_idx[i] = 0;
		recv_mem_deque_idx[i] = 0;
		for (int j = 0; j < MEM_CNT; j++)
		{
			send_mem_ptr[i][j] = NULL;
			recv_mem_ptr[i][j] = NULL;
			send_mem_len[i][j] = -1;
			recv_mem_len[i][j] = -1;
		}
		header_cache[i] = (char*)malloc(head_sz);

	}

}

//偶数是forward，奇数是backward
int launch_send_thread(int send_channel_id)
{
	pthread_t threadId = 0;
	int ret = pthread_create(&threadId, 0, &send_td, &send_channel_id);
	pthread_detach(threadId);
	sleep(3);
	return 0;
}
int launch_recv_thread(int recv_channel_id)
{
	pthread_t threadId = 0;
	int ret = pthread_create(&threadId, 0, &recv_td, &recv_channel_id);
	pthread_detach(threadId);
	sleep(3);
	return 0;
}

void* recv_td(void* args)
{
	int worker_id = *(int*)args;
	printf("recv worker_id =%d\n", worker_id);
	int ps_port_id = ps_id * worker_num + worker_id;
	int connfd = wait4connection(ps_ips[ps_id], to_ps_ports[ps_port_id] );
	printf("got connection connfd=%d\n", connfd);

	while (1 == 1)
	{
		int enque_idx = recv_mem_enque_idx[worker_id];
		if (recv_mem_len[worker_id][enque_idx] > 0)
		{
			//not free
		}
		else
		{
			char* sta_buf = header_cache[worker_id];
			recv_fixed_len(connfd, sta_buf, head_sz);
			struct Header* hdr = (struct Header*)(void*)sta_buf;
			size_t data_len = hdr->data_len;
			//printf("before alloc  recv_mem_ptr[worker_id][enque_idx]=%p\n", recv_mem_ptr[worker_id][enque_idx]);


			recv_mem_ptr[worker_id][enque_idx] = (char*)malloc(data_len);
			if (recv_mem_ptr[worker_id][enque_idx] == NULL)
			{
				printf("Alloc Fail recv mem\n");
				exit(-1);
			}
			sta_buf = recv_mem_ptr[worker_id][enque_idx];
			/*
			//if (data_len > 1000000)
			{
				printf("OK alloc recv_mem_ptr[%d][%d]=%p data_len=%ld\n", worker_id, enque_idx, sta_buf, data_len);
				for (int kk = 0; kk < data_len; kk++)
				{
					sta_buf[kk] = 'a';
					if (kk % 10000 == 0)
					{
						printf("kk=%d\n", kk );
					}
				}
				printf("test kk ok\n");
			}
			**/
			recv_fixed_len(connfd, sta_buf, data_len);
			recv_mem_len[worker_id][enque_idx] = data_len;
			//printf("recv ok worker_id=%d  enque_idx=%d data_len=%ld\n", worker_id, enque_idx, data_len);
			recv_mem_enque_idx[worker_id] = (recv_mem_enque_idx[worker_id] + 1) % MEM_CNT;
		}
	}
}

void* send_td(void* args)
{
	int to_worker_id = *(int*)args;
	printf("to_worker_id=%d\n", to_worker_id);

	int to_worker_port_id = to_worker_id + ps_id * worker_num;
	int send_fd = go4connection(worker_ips[to_worker_id], to_worker_ports[to_worker_port_id]);
	printf("Connected to %s  %d\n",  worker_ips[to_worker_id], to_worker_ports[to_worker_port_id]);
	while (1 == 1)
	{
		int deque_idx = send_mem_deque_idx[to_worker_id];
		if (send_mem_len[to_worker_id][deque_idx] < 0 )
		{
			//free slot, if comes here, no need to send, wait
		}
		else
		{
			//has filled with data, can be sent
			//TODO: Send Op  encode data_len, data
			char* sta_buf = send_mem_ptr[to_worker_id][deque_idx];
			size_t send_len = send_mem_len[to_worker_id][deque_idx] + head_sz;
			int ret = send(send_fd, sta_buf, send_len, 0);
			free(send_mem_ptr[to_worker_id][deque_idx]);
			send_mem_ptr[to_worker_id][deque_idx] = NULL;
			send_mem_len[to_worker_id][deque_idx] = -1;
			//printf("free %d\n", backward_send_deque_index);
			send_mem_deque_idx[to_worker_id] = (send_mem_deque_idx[to_worker_id] + 1) % MEM_CNT;
		}

	}

	return NULL;
}


int inquire_recv_mem_len(int worker_id)
{
	int deque_idx = recv_mem_deque_idx[worker_id];
	if (deque_idx < 0 || recv_mem_len[worker_id][deque_idx] < 0)
	{
		return -1;
	}
	else
	{
		return recv_mem_len[worker_id][deque_idx];
	}
}
int dequeue_recv_mem(int worker_id, char* data_out, int data_len)
{
	int deque_idx = recv_mem_deque_idx[worker_id];
	//printf("deque_idx=%d recv_mem_len[worker_id][deque_idx]=%d  data_len=%d\n", deque_idx, recv_mem_len[worker_id][deque_idx], data_len);
	if (deque_idx < 0 || recv_mem_len[worker_id][deque_idx] < 0)
	{
		return -1;
	}
	else
	{
		char* sta_idx = recv_mem_ptr[worker_id][deque_idx];
		//printf("Memcpy deque_idx=%d recv_mem_len[worker_id][deque_idx]=%d data_len=%d  eq_len=%d\n", deque_idx, recv_mem_len[worker_id][deque_idx], data_len, recv_mem_len[worker_id][deque_idx]);
		memcpy(data_out, sta_idx, data_len);
		//printf("free recv_mem_len[%d][%d]=%d data_len=%d sta_idx=%p\n", worker_id, deque_idx, recv_mem_len[worker_id][deque_idx], data_len, sta_idx);
		free(recv_mem_ptr[worker_id][deque_idx]);
		//printf("FIN free recv_mem_len[%d][%d]=%d data_len=%d\n", worker_id, deque_idx, recv_mem_len[worker_id][deque_idx], data_len);
		recv_mem_ptr[worker_id][deque_idx] = NULL;
		recv_mem_len[worker_id][deque_idx] = -1;
		//printf("work_id=%d deque_idx=%d data_len=%d\n", worker_id, deque_idx, data_len);
		recv_mem_deque_idx[worker_id] = (recv_mem_deque_idx[worker_id] + 1) % MEM_CNT;

		return 0;
	}
}

int enque_send_mem(int worker_id, char* data_in, int in_len)
{
	//加头封装
	int enque_idx = send_mem_enque_idx[worker_id];
	if (enque_idx < 0 || send_mem_len[worker_id][enque_idx] > 0 )
	{
		return -1;
	}
	else
	{
		//find a free slot, fill it, then mark it ready (not free)
		send_mem_ptr[worker_id][enque_idx] = (char*)malloc(in_len + head_sz);
		char* mem_sta = send_mem_ptr[worker_id][enque_idx];
		if (mem_sta == NULL)
		{
			printf("Alloc Mem Fail\n");
			exit(-1);
		}
		struct Header* hrd = (struct Header*)(void*)mem_sta;
		hrd->data_len = in_len;
		char* sta_idx = mem_sta + head_sz;
		memcpy(sta_idx, data_in, in_len);
		send_mem_len[worker_id][enque_idx] = in_len;
		send_mem_enque_idx[worker_id] = (send_mem_enque_idx[worker_id] + 1) % MEM_CNT;
	}
}


void recv_fixed_len(int connfd, char* sta_buf, int len)
{

	size_t recved_len = 0;
	size_t remained_len = len;
	int ret = 0;
	while (recved_len < len)
	{
		//printf("connfd=%d  recv_fixed_len  len=%d\n", connfd, len);
		ret = recv(connfd, sta_buf + recved_len, remained_len, 0);
		if (ret < 0)
		{
			printf("ret= %d connfd=%d  len=%d recved_len=%ld remained_len=%ld errno=%d \n", ret, connfd, len, recved_len, remained_len, errno);
			exit(-1);
		}
		else
		{
			recved_len += ret;
			remained_len -= ret;
		}
	}
	//printf("success ret= %d connfd=%d  len=%d recved_len=%ld remained_len=%ld \n", ret, connfd, len, recved_len, remained_len);
}



int wait4connection(char*local_ip, int local_port)
{
	int fd = socket(PF_INET, SOCK_STREAM , 0);
	struct sockaddr_in address;
	bzero(&address, sizeof(address));
	//转换成网络地址
	address.sin_port = htons(local_port);
	address.sin_family = AF_INET;
	//地址转换
	inet_pton(AF_INET, local_ip, &address.sin_addr);
	//绑定ip和端口
	int check_ret = -1;
	do
	{
		printf("binding...\n");
		check_ret = bind(fd, (struct sockaddr*)&address, sizeof(address));
		sleep(1);
	}
	while (check_ret >= 0);
	printf("bind ok\n");
	//创建监听队列，用来存放待处理的客户连接
	check_ret = listen(fd, 5);
	struct sockaddr_in addressClient;
	socklen_t clientLen = sizeof(addressClient);
	printf("listening at %s %d\n", local_ip, local_port );
	//接受连接，阻塞函数
	int connfd = accept(fd, (struct sockaddr*)&addressClient, &clientLen);
	return connfd;
}

int go4connection(char* remote_ip, int remote_port)
{
	printf("Connecting to %s:%d\n", remote_ip, remote_port);
	int fd;
	int check_ret;
	fd = socket(PF_INET, SOCK_STREAM, 0);
	assert(fd >= 0);
	struct sockaddr_in address;
	bzero(&address, sizeof(address));
	//转换成网络地址
	address.sin_port = htons(remote_port);
	address.sin_family = AF_INET;
	//地址转换
	inet_pton(AF_INET, remote_ip, &address.sin_addr);
	do
	{
		check_ret = connect(fd, (struct sockaddr*) &address, sizeof(address));
		sleep(1);
		printf("Connecting to %s:%d\n", remote_ip, remote_port);
	}
	while (check_ret < 0);
	//发送数据
	printf("connected to %s %d\n", remote_ip, remote_port);
	return fd;
}


int main()
{
	//for test
	int p_id = 0;
	int p_num = 1;
	int w_num = 2;
	init_info(p_id, p_num, w_num);
	init_mem();
	for (int work_id = 0; work_id < worker_num; work_id++)
	{
		int recv_channel_id = work_id;
		launch_recv_thread(recv_channel_id);
	}
	sleep(5);
	for (int work_id = 0; work_id < worker_num; work_id++)
	{
		int send_channel_id = work_id;
		launch_recv_thread(send_channel_id);

	}
	printf("Connection Initialized worker worker_num=%d\n", worker_num);
	getchar();

	while (1 == 1)
	{
		for (int work_id = 0; work_id < worker_num; work_id++)
		{
			int data_len = inquire_recv_mem_len(work_id);
			if (data_len > 0)
			{
				printf("worker_id=%d\t data_len=%d", work_id, data_len);
				char* data_out = (char*)malloc(data_len);
				int ret = dequeue_recv_mem(work_id, data_out, data_len);
				printf("data_len=%d ret=%d work_id=%d", data_len, ret, work_id);
			}
		}

	}


}



