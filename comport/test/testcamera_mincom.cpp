#include <iostream>
#include <cstdlib>
#include<stdio.h>
#include<string.h>
#include<stdlib.h>
#include<fcntl.h>
#include<sys/ioctl.h>
#include<unistd.h>
#include<time.h>
#include<termios.h>
#include<sys/select.h>
#include<sys/types.h>
using namespace std;

void init_ttyS(int fd)
{
	struct termios newtio;
	bzero(&newtio,sizeof(newtio));
	tcgetattr(fd,&newtio);
	cfsetispeed(&newtio,B115200);
	cfsetospeed(&newtio,B115200);
	newtio.c_cflag |=(CLOCAL|CREAD);
	newtio.c_cflag &=~PARENB;
	newtio.c_cflag &=~CSTOPB;
	newtio.c_cflag &=~CSIZE;
	newtio.c_cflag |=CS8;
	newtio.c_lflag &=~(ICANON|ECHO|ECHOE|ISIG);
	newtio.c_oflag &=~OPOST;
	newtio.c_cc[VTIME] =0;
	newtio.c_cc[VMIN] =0;
	tcflush(fd,TCIFLUSH);
	tcsetattr(fd,TCSANOW,&newtio);
	}
int main(int argn,char** argv){
	//system("/bin/sh ./detectscript.sh");
	int t,fd,ret;
	unsigned char wr_buf[256]="GOOD BOY";
	if(argn >= 7){
		wr_buf[0] = 0xff;
		wr_buf[1] = 0x00;
		wr_buf[2] = 0x09;
		wr_buf[3] = 0x05;
		wr_buf[4] =  (unsigned char)atoi(argv[1]);
		wr_buf[5] =  (unsigned char)atoi(argv[2]);
		wr_buf[8] = (unsigned char)atoi(argv[3]);
		wr_buf[9] = (unsigned char)atoi(argv[4]);
		wr_buf[10] = (unsigned char)atoi(argv[5]);
		wr_buf[11] = (unsigned char)atoi(argv[6]);
		wr_buf[6] = (unsigned char)((wr_buf[8]+wr_buf[10])/2);
		wr_buf[7] = (unsigned char)((wr_buf[9]+wr_buf[11])/2);
		wr_buf[12]	= 0xA0;
	}
	wr_buf[13] = wr_buf[0];
	for (t = 1;t<12;t++){
		wr_buf[13] ^= wr_buf[t];
	}
	wr_buf[12] = wr_buf[13];
	char device[]="/dev/ttyS0";
	printf("\n UART Working\n");
	fd=open(device,O_RDWR);
	if(fd<0)
	{
		printf("open uart0 failed");
		return -1;
	}
	 else 
	 printf("\n success open uart0\n");
	init_ttyS(fd);
	while(1){
	ret=write(fd,wr_buf,13);
	printf("SEND DATA\n");
	usleep(1000000);
	}
	close(fd);
	return 0;
}
