#include "OrthographicCamera.h"

OrthographicCamera::OrthographicCamera() : SimpleCamera() {}

OrthographicCamera::~OrthographicCamera() {}

void OrthographicCamera::setOrthoBounds(float left, float right, float bottom, float top)
{
    m_left = left;
    m_right = right;
    m_bottom = bottom;
    m_top = top;
}

void OrthographicCamera::getCameraProjectionMatrix(float projectionMatrix[16]) const
{
    float nearVal = getCameraFrustumNear();
    float farVal = getCameraFrustumFar();

    projectionMatrix[0] = 2.0f / (m_right - m_left);
    projectionMatrix[1] = 0.0f;
    projectionMatrix[2] = 0.0f;
    projectionMatrix[3] = 0.0f;

    projectionMatrix[4] = 0.0f;
    projectionMatrix[5] = 2.0f / (m_top - m_bottom);
    projectionMatrix[6] = 0.0f;
    projectionMatrix[7] = 0.0f;

    projectionMatrix[8] = 0.0f;
    projectionMatrix[9] = 0.0f;
    projectionMatrix[10] = -2.0f / (farVal - nearVal);
    projectionMatrix[11] = 0.0f;

    projectionMatrix[12] = -(m_right + m_left) / (m_right - m_left);
    projectionMatrix[13] = -(m_top + m_bottom) / (m_top - m_bottom);
    projectionMatrix[14] = -(farVal + nearVal) / (farVal - nearVal);
    projectionMatrix[15] = 1.0f;
}
