#include "stm32f10x.h"                  // Device header
//A1-PIN_1---level;;PIN_2--- vertical
void PWM_Init(void)
{
	RCC_APB1PeriphClockCmd(RCC_APB1Periph_TIM2, ENABLE);
	RCC_APB2PeriphClockCmd(RCC_APB2Periph_GPIOA, ENABLE);
	
	GPIO_InitTypeDef GPIO_L_InitStructure;
	GPIO_L_InitStructure.GPIO_Mode = GPIO_Mode_AF_PP;
	GPIO_L_InitStructure.GPIO_Pin = GPIO_Pin_1 ;//PIN_1---level;;PIN_2---- vertical
	GPIO_L_InitStructure.GPIO_Speed = GPIO_Speed_50MHz;
	GPIO_Init(GPIOA, &GPIO_L_InitStructure);
	
	TIM_InternalClockConfig(TIM2);
	
	TIM_TimeBaseInitTypeDef TIM_L_TimeBaseInitStructure;
	TIM_L_TimeBaseInitStructure.TIM_ClockDivision = TIM_CKD_DIV1;
	TIM_L_TimeBaseInitStructure.TIM_CounterMode = TIM_CounterMode_Up;
	TIM_L_TimeBaseInitStructure.TIM_Period = 20000 - 1;		//ARR
	TIM_L_TimeBaseInitStructure.TIM_Prescaler = 72 - 1;		//PSC
	TIM_L_TimeBaseInitStructure.TIM_RepetitionCounter = 0;
	TIM_TimeBaseInit(TIM2, &TIM_L_TimeBaseInitStructure);
	
	TIM_OCInitTypeDef TIM_L_OCInitStructure;
	TIM_OCStructInit(&TIM_L_OCInitStructure);
	TIM_L_OCInitStructure.TIM_OCMode = TIM_OCMode_PWM1;
	TIM_L_OCInitStructure.TIM_OCPolarity = TIM_OCPolarity_High;
	TIM_L_OCInitStructure.TIM_OutputState = TIM_OutputState_Enable;
	TIM_L_OCInitStructure.TIM_Pulse = 0;		//CCR
	TIM_OC2Init(TIM2, &TIM_L_OCInitStructure);
	
	TIM_Cmd(TIM2, ENABLE);
	
	
	RCC_APB1PeriphClockCmd(RCC_APB1Periph_TIM3, ENABLE);
	RCC_APB2PeriphClockCmd(RCC_APB2Periph_GPIOB, ENABLE);
	
	GPIO_InitTypeDef GPIO_V_InitStructure;
	GPIO_V_InitStructure.GPIO_Mode = GPIO_Mode_AF_PP;
	GPIO_V_InitStructure.GPIO_Pin = GPIO_Pin_0 ;//PIN_1---level;;PIN_2---- vertical
	GPIO_V_InitStructure.GPIO_Speed = GPIO_Speed_50MHz;
	GPIO_Init(GPIOB, &GPIO_V_InitStructure);
	
	TIM_InternalClockConfig(TIM3);
	
	TIM_TimeBaseInitTypeDef TIM_V_TimeBaseInitStructure;
	TIM_V_TimeBaseInitStructure.TIM_ClockDivision = TIM_CKD_DIV1;
	TIM_V_TimeBaseInitStructure.TIM_CounterMode = TIM_CounterMode_Up;
	TIM_V_TimeBaseInitStructure.TIM_Period = 20000 - 1;		//ARR
	TIM_V_TimeBaseInitStructure.TIM_Prescaler = 72 - 1;		//PSC
	TIM_V_TimeBaseInitStructure.TIM_RepetitionCounter = 0;
	TIM_TimeBaseInit(TIM3, &TIM_V_TimeBaseInitStructure);
	
	TIM_OCInitTypeDef TIM_V_OCInitStructure;
	TIM_OCStructInit(&TIM_V_OCInitStructure);
	TIM_V_OCInitStructure.TIM_OCMode = TIM_OCMode_PWM1;
	TIM_V_OCInitStructure.TIM_OCPolarity = TIM_OCPolarity_High;
	TIM_V_OCInitStructure.TIM_OutputState = TIM_OutputState_Enable;
	TIM_V_OCInitStructure.TIM_Pulse = 0;		//CCR
	TIM_OC3Init(TIM3, &TIM_V_OCInitStructure);
	
	TIM_Cmd(TIM3, ENABLE);
}

