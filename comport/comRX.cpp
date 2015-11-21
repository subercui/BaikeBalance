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
	cfsetospeed(&newtio,B115200);// above 3 lines, set boltrate
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
	int t,fd,nread;
	unsigned char buff[256];
	char device[]="/dev/ttyS0";
	printf("\n UART Working\n");
	fd=open(device,O_RDONLY);
	if(fd<0)
	{
		printf("open uart0 failed");
		return -1;
	}
	 else 
	printf("\n success open uart0\n");
	init_ttyS(fd);
	while(1){
	nread=read(fd,buff,13);
    buff[nread]='\0';
	printf("SEND DATA\n");
	printf("%s\n",buff);
	usleep(10);
	}
	close(fd);
	return 0;
}
