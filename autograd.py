# TORCH.AUTOGRAD를 사용한 자동 미분
# 신경망을 학습할 때 가장 자주 사용되는 알고리즘은 역전파입니다. 
# 이 알고리즘에서, 매개변수(모델 가중치)는 주어진 
# 매개변수에 대한 손실 함수의 변화도(gradient)에 따라 조정됩니다.
# 이러한 변화도를 계산하기 위해 PyTorch에는 
# torch.autograd라고 불리는 자동 미분 엔진이 내장되어 있습니다. 
# 이는 모든 계산 그래프에 대한 변화도의 자동 계산을 지원합니다.
# 입력 x, 매개변수 w와 b , 그리고 일부 손실 함수가 있는 
# 가장 간단한 단일 계층 신경망을 가정하겠습니다. 
# PyTorch에서는 다음과 같이 정의할 수 있습니다:
import torch

x = torch.ones(5)  # input tensor
y = torch.zeros(3)  # expected output
w = torch.randn(5, 3, requires_grad=True)
b = torch.randn(3, requires_grad=True)
z = torch.matmul(x, w)+b
loss = torch.nn.functional.binary_cross_entropy_with_logits(z, y)

# 이 신경망에서, w와 b는 최적화를 해야 하는 매개변수입니다. 
# 따라서 이러한 변수들에 대한 손실 함수의 변화도를 계산할 수 있어야 합니다. 
# 이를 위해서 해당 텐서에 requires_grad 속성을 설정합니다.

# 연산 그래프를 구성하기 위해 텐서에 적용하는 함수는 사실 Function 클래스의 객체입니다. 
# 이 객체는 순전파 방향으로 함수를 계산하는 방법과, 
# 역방향 전파 단계에서 도함수(derivative)를 계산하는 방법을 알고 있습니다. 
# 역방향 전파 함수에 대한 참조(reference)는 텐서의 grad_fn 속성에 저장됩니다. 
# Function에 대한 자세한 정보는 이 문서 에서 찾아볼 수 있습니다.
print('Gradient function for z =', z.grad_fn)
print('Gradient function for loss =', loss.grad_fn)

# 변화도(Gradient) 계산하기
# 신경망에서 매개변수의 가중치를 최적화하려면 
# 매개변수에 대한 손실함수의 도함수(derivative)를 계산해야 합니다. 
# 이러한 도함수를 계산하기 위해, 
# loss.backward() 를 호출한 다음 w.grad와 b.grad에서 값을 가져옵니다
loss.backward()
print(w.grad)
print(b.grad)

# 변화도 추적 멈추기
# 기본적으로, requires_grad=True인 모든 텐서들은 
# 연산 기록을 추적하고 변화도 계산을 지원합니다. 
# 그러나 모델을 학습한 뒤 입력 데이터를 단순히 
# 적용하기만 하는 경우와 같이 순전파 연산만 필요한 경우에는, 
# 이러한 추적이나 지원이 필요없을 수 있습니다. 
# 연산 코드를 torch.no_grad() 블록으로 둘러싸서 연산 추적을 멈출 수 있습니다:
z = torch.matmul(x, w)+b
print(z.requires_grad)

with torch.no_grad():
    z = torch.matmul(x, w)+b
print(z.requires_grad)

# 동일한 결과를 얻는 다른 방법은 텐서에 detach() 메소드를 사용하는 것입니다:
z = torch.matmul(x, w)+b
z_det = z.detach()
print(z_det.requires_grad)

# 변화도 추적을 멈춰야 하는 이유들은 다음과 같습니다.
# 신경망의 일부 매개변수를 고정된 매개변수(frozen parameter)로 표시합니다. 
# 이는 사전 학습된 신경망을 미세조정 할 때 매우 일반적인 시나리오입니다.
# 변화도를 추적하지 않는 텐서의 연산이 더 효율적이기 때문에, 
# 순전파 단계만 수행할 때 연산 속도가 향상됩니다.