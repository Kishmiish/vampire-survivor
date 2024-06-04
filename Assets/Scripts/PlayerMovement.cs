using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class PlayerMovement : MonoBehaviour
{
    public float speed;
    
    [HideInInspector]
    public float lastHorizontalVector;

    [HideInInspector]
    public float lastVerticalVector;

    [HideInInspector]
    public Rigidbody2D rb;

    [HideInInspector]
    public Vector2 moveDirection;

    void Start()
    {
        rb = GetComponent<Rigidbody2D>();
    }

    void Update()
    {
        InputManegement();  
    }
    void FixedUpdate()
    {
        Move();
    }

    void InputManegement()
    {
        float mx = Input.GetAxisRaw("Horizontal");
        float my = Input.GetAxisRaw("Vertical");

        moveDirection = new Vector2(mx, my).normalized;

        if (moveDirection.x != 0) { lastHorizontalVector = moveDirection.x; }
        if (moveDirection.y != 0) { lastVerticalVector = moveDirection.y; }
    }

    void Move()
    {
        transform.Translate(moveDirection.x * speed, moveDirection.y * speed, 0);
        //rb.velocity = new Vector2(moveDirection.x * speed, moveDirection.y * speed);
    }
}