#include "stm32f10x.h"                  // Device header
#include "PWM.h"

extern float Level_Angle;
extern float Vertical_Angle;

void Servo_Init(void)
{
	PWM_Init();
}

void Servo_Angle_Control(char* direction, uint16_t Angle){
	if(Angle>270) 
{Angle=270;}
	if(Angle<0) 
{Angle=0;}
	if(direction[0] == 'l'){
		TIM_SetCompare2(TIM2, 2000 * Angle/270 + 500);
	}
	else{
		TIM_SetCompare3(TIM3, 2000 * Angle/270 + 500);
	}
}

/*
4---�Ƿ�����ת
3---�Ƿ�����ת
2---�Ƿ���ת
1---�Ƿ���ת

��ת----0000 101X---0x0A
��ת----0000 100X---0x08
��ת----0000 01X1---0x05
��ת----0000 01X0---0x04

*/

void Servo_COM_Control(uint8_t HEX)
{
	if(HEX >= 0x0C )
	{
		return;
	}
	if(HEX & 0x08){
		if(HEX & 0x02){
			Level_Angle-=5;
		}
		else{
			Level_Angle+=5;
		}
	}
	if(HEX & 0x04){
		if(HEX & 0x01){
			Vertical_Angle+=5;
		}
		else{
			Vertical_Angle-=5;
		}
	}
}
