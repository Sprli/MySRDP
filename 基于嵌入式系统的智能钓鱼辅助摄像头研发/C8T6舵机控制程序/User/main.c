#include "stm32f10x.h"                  // Device header
#include "Delay.h"
#include "OLED.h"
#include "Servo.h"
#include "Serial.h"
#include "PWM.h"

uint8_t RxData;
uint8_t KeyNum;
float Level_Angle;
float Vertical_Angle;

int main(void)
{
	Serial_Init();
	OLED_Init();
	Servo_Init();
	level_angle = 135;
	vertical_angle = 0;
	while(1)
	{
		OLED_ShowString(1, 1, "Serial_Info:");
		OLED_ShowString(2, 1, "Cur_lev:");
		OLED_ShowString(3, 1, "Cur_ver:");
		OLED_ShowNum(1, 13, Serial_RxFlag, 1);
		OLED_ShowSignedNum(2, 9, level_angle, 4);
		OLED_ShowSignedNum(3, 9, vertical_angle, 4);
		if(Serial_RxFlag == 1){  // 开始控制舵机
//			Servo_Angle_Control("level", level_angle);
//			Servo_Angle_Control("vertical", vertical_angle);
			Serial_RxFlag = 0;
		}
		Servo_Angle_Control("level", level_angle);
		Servo_Angle_Control("vertical", vertical_angle);
	}
	
}
