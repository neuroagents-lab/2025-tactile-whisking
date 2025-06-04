#ifndef ORTHOGRAPHIC_CAMERA_H
#define ORTHOGRAPHIC_CAMERA_H

#include "SimpleCamera.h"

struct OrthographicCamera : public SimpleCamera
{
    OrthographicCamera();
    virtual ~OrthographicCamera();

    void setOrthoBounds(float left, float right, float bottom, float top);
    virtual void getCameraProjectionMatrix(float projectionMatrix[16]) const override;

private:
    float m_left = -10.0f;
    float m_right = 10.0f;
    float m_bottom = -10.0f;
    float m_top = 10.0f;
};

#endif // ORTHOGRAPHIC_CAMERA_H
