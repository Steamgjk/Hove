//gcc -shared -o glue.so -fPIC glue.c -lpthread
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
#define MEM_CNT (5)
#define MEM_UNIT_SIZE (1024*1024*100)
#define BUFFER_SIZE (MEM_CNT*MEM_UNIT_SIZE) //500MB
#define LABEL_LEN (32)
#define CAP 128

char* worker_ips[CAP] = {"127.0.0.1", "127.0.0.1", "127.0.0.1", "127.0.0.1", \
                         "127.0.0.1", "127.0.0.1", "127.0.0.1", "127.0.0.1"
                        };
int worker_ports[CAP] = {5510, 5511, 5512, 5513, 5514, 5515, 5516, 5517};
int worker_id = 0;
int worker_num = 4;

char* forward_send_mem_ptr = NULL;
int forward_send_mem_len[MEM_CNT];
int forward_send_enque_index = -1;
int forward_send_deque_index = -1;

char* backward_send_mem_ptr = NULL;
int backward_send_mem_len[MEM_CNT];
int backward_send_enque_index = -1;
int backward_send_deque_index = -1;


char* forward_recv_mem_ptr = NULL;
int forward_recv_mem_len[MEM_CNT];
int forward_recv_enque_index = -1;
int forward_recv_deque_index = -1;

char* backward_recv_mem_ptr = NULL;
int backward_recv_mem_len[MEM_CNT];
int backward_recv_enque_index = -1;
int backward_recv_deque_index = -1;


int init_worker_info(int wid, int wn);
int init_mem();
int enque_forward_send_mem(char* data_in, int in_len);
int enque_backward_send_mem(char* data_in, int in_len);
int inquire_forward_recv_mem_len();
int dequeue_forward_recv_mem(char* data_out, int data_len);
int inquire_backward_recv_mem_len();
int dequeue_backward_recv_mem(char* data_out, int data_len);
void* recv_td(void* args);
void* send_td(void* args);
int launch_send_thread(int thread_id);
int launch_recv_thread(int thread_id);

void recv_fixed_len(int connfd, char* sta_buf, int len);
int wait4connection(char*local_ip, int local_port);
int go4connection(int send_thread_id);

struct Header
{
	size_t data_len;
};
const int head_sz = sizeof(struct Header);

int init_worker_info(int wid, int wn)
{
	worker_id = wid;
	worker_num = wn;
	return 0;
}
int init_mem()
{
	forward_send_mem_ptr = (char*)malloc(BUFFER_SIZE);
	forward_recv_mem_ptr = (char*)malloc(BUFFER_SIZE);
	backward_send_mem_ptr = (char*)malloc(BUFFER_SIZE);
	backward_recv_mem_ptr = (char*)malloc(BUFFER_SIZE);
	if (forward_send_mem_ptr == NULL || forward_recv_mem_ptr == NULL || backward_send_mem_ptr == NULL || backward_recv_mem_ptr == NULL)
	{
		return -1;
	}
	else
	{
		forward_send_enque_index = 0;
		forward_send_deque_index = 0;
		forward_recv_enque_index = 0;
		forward_recv_deque_index = 0;
		backward_send_enque_index = 0;
		backward_send_deque_index = 0;
		backward_recv_enque_index = 0;
		backward_recv_deque_index = 0;
		for (int i = 0; i < MEM_CNT; i++)
		{
			forward_send_mem_len[i] = -1;
			forward_recv_mem_len[i] = -1;
			backward_send_mem_len[i] = -1;
			backward_recv_mem_len[i] = -1;
		}
		return 0;
	}
}
int enque_forward_send_mem(char* data_in, int in_len)
{
	//加头封装

	if (forward_send_enque_index < 0 || forward_send_mem_len[forward_send_enque_index] > 0 )
	{
		return -1;
	}
	else
	{
		printf("forward enque idx %d\n", forward_send_enque_index);
		//find a free slot, fill it, then mark it ready (not free)
		char* mem_sta = forward_send_mem_ptr + forward_send_enque_index * MEM_UNIT_SIZE;
		struct Header* hrd = (struct Header*)(void*)mem_sta;
		hrd->data_len = in_len;
		char* sta_idx = mem_sta + head_sz;
		memcpy(sta_idx, data_in, in_len);
		forward_send_mem_len[forward_send_enque_index] = in_len;
		//printf("forward_send_mem_len[%d]=%d\n", forward_send_enque_index, in_len );
		forward_send_enque_index = (forward_send_enque_index + 1) % MEM_CNT;
		return 0;
	}
}

int enque_backward_send_mem(char* data_in, int in_len)
{
	//加头封装
	printf("backward enque idx %d\n", backward_send_enque_index);
	if (backward_send_enque_index < 0 || backward_send_mem_len[backward_send_enque_index] > 0 )
	{
		return -1;
	}
	else
	{
		//find a free slot, fill it, then mark it ready (not free)
		char* mem_sta = backward_send_mem_ptr + backward_send_enque_index * MEM_UNIT_SIZE;
		struct Header* hrd = (struct Header*)(void*)mem_sta;
		hrd->data_len = in_len;
		char* sta_idx = mem_sta + head_sz;
		memcpy(sta_idx, data_in, in_len);
		backward_send_mem_len[backward_send_enque_index] = in_len;
		printf("backward_send_enque_index=%d  backward_send_mem_len=%d\n", backward_send_enque_index, backward_send_mem_len[backward_send_enque_index] );
		backward_send_enque_index = (backward_send_enque_index + 1) % MEM_CNT;
		return 0;
	}
}




int inquire_forward_recv_mem_len()
{
	if (forward_recv_deque_index < 0 || forward_recv_mem_len[forward_recv_deque_index] < 0)
	{
		return -1;
	}
	else
	{
		return forward_recv_mem_len[forward_recv_deque_index] ;
	}
}
int dequeue_forward_recv_mem(char* data_out, int data_len)
{
	if (forward_recv_deque_index < 0 || forward_recv_mem_len[forward_recv_deque_index] < 0)
	{
		return -1;
	}
	else
	{
		char* sta_idx = forward_recv_mem_ptr + (forward_recv_deque_index * MEM_UNIT_SIZE);
		sta_idx = sta_idx + head_sz;
		memcpy(data_out, sta_idx, data_len);
		forward_recv_mem_len[forward_recv_deque_index] = -1;
		forward_recv_deque_index = (forward_recv_deque_index + 1) % MEM_CNT;
		return 0;
	}
}

int inquire_backward_recv_mem_len()
{
	if (backward_recv_deque_index < 0 || backward_recv_mem_len[backward_recv_deque_index] < 0)
	{
		return -1;
	}
	else
	{
		return backward_recv_mem_len[backward_recv_deque_index] ;
	}
}
int dequeue_backward_recv_mem(char* data_out, int data_len)
{
	if (backward_recv_deque_index < 0 || backward_recv_mem_len[backward_recv_deque_index] < 0)
	{
		return -1;
	}
	else
	{
		char* sta_idx = backward_recv_mem_ptr + (backward_recv_deque_index * MEM_UNIT_SIZE);
		sta_idx = sta_idx + head_sz;
		memcpy(data_out, sta_idx, data_len);
		backward_recv_mem_len[backward_recv_deque_index] = -1;
		backward_recv_deque_index = (backward_recv_deque_index + 1) % MEM_CNT;
		return 0;
	}
}



//偶数是forward，奇数是backward
int launch_send_thread(int thread_id)
{
	pthread_t threadId = 0;
	int ret = pthread_create(&threadId, 0, &send_td, &thread_id);
	pthread_detach(threadId);
	sleep(3);
	return 0;
}
int launch_recv_thread(int thread_id)
{
	pthread_t threadId = 0;
	int ret = pthread_create(&threadId, 0, &recv_td, &thread_id);
	pthread_detach(threadId);
	sleep(3);
	return 0;
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
			printf("ret= %d connfd=%d\n", ret, connfd);
			exit(-1);
		}
		else
		{
			recved_len += ret;
			remained_len -= ret;
		}
	}
}
void* recv_td(void* args)
{
	int recv_thread_id = *(int*)args;
	printf("recv_thread_id =%d\n", recv_thread_id);
	int is_forward_recv = 1;
	if (recv_thread_id % 2 == 1)
	{
		//backward
		int connfd = wait4connection(worker_ips[recv_thread_id], worker_ports[recv_thread_id] );
		printf("got backward connection connfd=%d\n", connfd);
		while (1 == 1)
		{
			if (backward_recv_mem_len[backward_recv_enque_index] > 0)
			{
				//not free
			}
			else
			{
				char* sta_buf = backward_recv_mem_ptr + backward_recv_enque_index * MEM_UNIT_SIZE;
				recv_fixed_len(connfd, sta_buf, head_sz);
				struct Header* hdr = (struct Header*)(void*)sta_buf;
				size_t data_len = hdr->data_len;
				sta_buf = sta_buf + head_sz;
				recv_fixed_len(connfd, sta_buf, data_len);
				backward_recv_mem_len[backward_recv_enque_index] = data_len + head_sz;
				backward_recv_enque_index = (backward_recv_enque_index + 1) % MEM_CNT;
			}
		}
	}
	else
	{
		int connfd = wait4connection(worker_ips[recv_thread_id], worker_ports[recv_thread_id] );
		printf("got forward connection connfd=%d\n", connfd);
		while (1 == 1)
		{
			if (forward_recv_mem_len[forward_recv_enque_index] > 0)
			{
				//not free
				printf("not free for  %d", forward_recv_mem_len[forward_recv_enque_index]);
			}
			else
			{

				char* sta_buf = forward_recv_mem_ptr + forward_recv_enque_index * MEM_UNIT_SIZE;
				//printf("FOrward Recving... header\n");
				recv_fixed_len(connfd, sta_buf, head_sz);
				//printf("Recved a header\n");
				struct Header* hdr = (struct Header*)(void*)sta_buf;
				size_t data_len = hdr->data_len;
				sta_buf = sta_buf + head_sz;
				recv_fixed_len(connfd, sta_buf, data_len);
				printf("FOrward Recved... data_len=%ld\n", data_len);
				forward_recv_mem_len[forward_recv_enque_index] = data_len + head_sz;
				forward_recv_enque_index = (forward_recv_enque_index + 1) % MEM_CNT;
			}
		}
	}

}

void* send_td(void* args)
{
	int send_thread_id = *(int*)args;
	printf("send_thread_id=%d\n", send_thread_id);
	int is_forward_recv = 1;
	if (send_thread_id % 2 == 1)
	{
		int send_fd = go4connection(send_thread_id);
		//backward
		while (1 == 1)
		{
			//printf("send func  backward_send_deque_index=%d  len=%d\n", backward_send_deque_index, backward_send_mem_len[backward_send_deque_index]);
			//sleep(1);
			if (backward_send_mem_len[backward_send_deque_index] < 0 )
			{
				//free slot, if comes here, no need to send, wait
			}
			else
			{
				//has filled with data, can be sent
				//TODO: Send Op  encode data_len, data
				char* sta_buf = backward_send_mem_ptr + backward_send_deque_index * MEM_UNIT_SIZE;
				size_t send_len = backward_send_mem_len[backward_send_deque_index] + head_sz;
				printf("Backward send...\n");
				int ret = send(send_fd, sta_buf, send_len, 0);

				backward_send_mem_len[backward_send_deque_index] = -1;
				printf("free %d\n", backward_send_deque_index);
				backward_send_deque_index = (backward_send_deque_index + 1) % MEM_CNT;
			}

		}

	}
	else
	{
		int send_fd = go4connection(send_thread_id);
		while (1 == 1)
		{
			//printf("send func\n");
			if (forward_send_mem_len[forward_send_deque_index] < 0 )
			{
				//free slot, if comes here, no need to send, wait
				//printf("send forward_send_mem_len[%d]=%d\n", forward_send_deque_index, forward_send_mem_len[forward_send_deque_index]);
				//sleep(1);
			}
			else
			{
				//has filled with data, can be sent
				//TODO: Send Op  encode data_len, data
				char* sta_buf = forward_send_mem_ptr + forward_send_deque_index * MEM_UNIT_SIZE;
				size_t send_len = forward_send_mem_len[forward_send_deque_index] + head_sz;
				printf("forward send len=%ld \n ", send_len);
				int ret = send(send_fd, sta_buf, send_len, 0);
				//int ret = send(send_fd, sta_buf, head_sz, 0);
				forward_send_mem_len[forward_send_deque_index] = -1;
				printf("ret=%d free %d\n", ret, forward_send_deque_index);
				forward_send_deque_index = (forward_send_deque_index + 1) % MEM_CNT;

			}

		}
	}

	return NULL;
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

int go4connection(int send_thread_id)
{
	printf("go4connection: send_thread_id=%d\n", send_thread_id);
	int remote_id = -1;
	if (send_thread_id % 2 == 0)
	{
		//forward thread send, connect to successor
		remote_id = send_thread_id + 2;
		if (remote_id >= worker_num * 2)
		{
			return -1;
		}
	}
	else
	{
		//backward thread  send to predecessor
		remote_id = send_thread_id - 2;
		if (remote_id <= 0)
		{
			return -1;
		}
	}
	char* remote_ip = worker_ips[remote_id];
	int remote_port = worker_ports[remote_id];
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
	init_mem();
}
