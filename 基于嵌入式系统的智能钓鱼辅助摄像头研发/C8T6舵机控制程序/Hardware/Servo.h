#ifndef __SERVO_H
#define __SERVO_H

void Servo_Init(void);
void Servo_Angle_Control(char* direction, uint16_t Angle);
void Servo_COM_Control(uint8_t HEX);

#endif
