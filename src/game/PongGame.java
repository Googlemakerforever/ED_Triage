package game;

import objects.Ball;
import objects.Paddle;

import java.awt.*;
import java.awt.event.KeyEvent;

/**
 * PongGame is a concrete implementation of the GDV5 framework for a classic 2-player Pong game.
 * <p>
 * Controls:
 * Player 1: W/S keys
 * Player 2: Up/Down arrow keys
 * First player to reach WINNING_SCORE wins. Press 'R' to restart after game over.
 */
public class PongGame extends GDV5 {
    // Game objects
    private Paddle leftPaddle; // Player 1 paddle
    private Paddle rightPaddle; // Player 2 paddle
    private Ball ball; // Ball object

    // Scores for each player
    private int leftScore = 0;
    private int rightScore = 0;

    // Game settings
    private static final int PADDLE_WIDTH = 15;
    private static final int PADDLE_HEIGHT = 80;
    private static final int PADDLE_SPEED = 5;
    private static final int BALL_SIZE = 15;
    private static final int WINNING_SCORE = 10;

    // Game state
    private boolean gameOver = false;
    private String winner = "";

    /**
     * Constructs a PongGame instance and initializes game objects.
     */
    public PongGame() {
        super(100); // 60 FPS
        setTitle("Two Player Pong");
        initializeGame();
    }

    /**
     * Initializes paddles and ball to their starting positions.
     */
    private void initializeGame() {
        // Create paddles at left and right sides, centered vertically
        leftPaddle = new Paddle(30, getMaxWindowY() / 2 - PADDLE_HEIGHT / 2, PADDLE_WIDTH, PADDLE_HEIGHT, PADDLE_SPEED);
        rightPaddle = new Paddle(getMaxWindowX() - 30 - PADDLE_WIDTH, getMaxWindowY() / 2 - PADDLE_HEIGHT / 2, PADDLE_WIDTH, PADDLE_HEIGHT, PADDLE_SPEED);

        // Create ball in center
        ball = new Ball(getMaxWindowX() / 2 - BALL_SIZE / 2,
                getMaxWindowY() / 2 - BALL_SIZE / 2,
                BALL_SIZE);
    }

    /**
     * Main game update loop. Handles input, movement, collision, scoring, and game over logic.
     */
    @Override
    public void update() {
        if (gameOver) {
            // Press R to restart
            if (KeysTyped[KeyEvent.VK_R]) {
                restartGame();
                KeysTyped[KeyEvent.VK_R] = false;
            }
            return;
        }

        // Player 1 controls (W/S)
        if (KeysPressed[KeyEvent.VK_W]) {
            leftPaddle.moveUp();
        }
        if (KeysPressed[KeyEvent.VK_S]) {
            leftPaddle.moveDown();
        }

        // Player 2 controls (UP/DOWN arrows)
        if (KeysPressed[KeyEvent.VK_UP]) {
            rightPaddle.moveUp();
        }
        if (KeysPressed[KeyEvent.VK_DOWN]) {
            rightPaddle.moveDown();
        }

        // Keep paddles in bounds of the window
        leftPaddle.stayInBounds(0, getMaxWindowY());
        rightPaddle.stayInBounds(0, getMaxWindowY());

        // Update ball position and speed
        ball.update();

        // Ball collision with top/bottom walls
        if (ball.getY() <= 0 || ball.getY() + ball.getSize() >= getMaxWindowY()) {
            ball.reverseY(); // Bounce off wall
        }

        // Ball collision with paddles
        Rectangle ballRect = ball.getRectangle();
        Rectangle leftPaddleRect = leftPaddle.getRectangle();
        Rectangle rightPaddleRect = rightPaddle.getRectangle();

        // Check collision with left paddle
        if (ballRect.intersects(leftPaddleRect)) {
            ball.reverseX(); // Bounce horizontally
            ball.setX(leftPaddle.getX() + leftPaddle.getWidth() + 1); // Prevent ball sticking to paddle
            addSpin(leftPaddle, ball); // Add spin based on hit position
        }

        // Check collision with right paddle
        if (ballRect.intersects(rightPaddleRect)) {
            ball.reverseX(); // Bounce horizontally
            ball.setX(rightPaddle.getX() - ball.getSize() - 1); // Prevent ball sticking to paddle
            addSpin(rightPaddle, ball); // Add spin based on hit position
        }

        // Scoring - ball goes off screen
        if (ball.getX() < -ball.getSize()) {
            // Right player scores
            rightScore++;
            resetBall(1); // Ball goes toward left player
            checkGameOver();
        } else if (ball.getX() > getMaxWindowX()) {
            // Left player scores
            leftScore++;
            resetBall(-1); // Ball goes toward right player
            checkGameOver();
        }
    }

    /**
     * Adds spin to the ball based on where it hits the paddle.
     * @param paddle The paddle the ball collided with
     * @param ball The ball object
     */
    private void addSpin(Paddle paddle, Ball ball) {
        // Calculate relative position where ball hit paddle (0.0 to 1.0)
        double relativeHitPos = (ball.getCenterY() - paddle.getY()) / paddle.getHeight();
        // Convert to range -1.0 to 1.0
        double spinFactor = (relativeHitPos - 0.5) * 2.0;
        // Apply spin to ball's Y velocity
        ball.adjustYVelocity(spinFactor * 2);
    }

    /**
     * Resets the ball to the center and sets its direction.
     * @param direction Direction for ball to move (-1 for left, 1 for right)
     */
    private void resetBall(int direction) {
        ball.reset(getMaxWindowX() / 2 - BALL_SIZE / 2,
                getMaxWindowY() / 2 - BALL_SIZE / 2,
                direction);
    }

    /**
     * Checks if either player has reached the winning score and sets game over state.
     */
    private void checkGameOver() {
        if (leftScore >= WINNING_SCORE) {
            gameOver = true;
            winner = "Player 1 Wins!";
        } else if (rightScore >= WINNING_SCORE) {
            gameOver = true;
            winner = "Player 2 Wins!";
        }
    }

    /**
     * Restarts the game by resetting scores, state, and reinitializing objects.
     */
    private void restartGame() {
        leftScore = 0;
        rightScore = 0;
        gameOver = false;
        winner = "";
        initializeGame();
    }

    /**
     * Renders all game graphics, including paddles, ball, scores, controls, and game over screen.
     * @param g The Graphics2D context to draw on
     */
    @Override
    public void draw(Graphics2D g) {
        // Enable anti-aliasing for smoother graphics
        g.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON);

        // Draw center dashed line
        g.setColor(Color.WHITE);
        g.setStroke(new BasicStroke(2, BasicStroke.CAP_ROUND, BasicStroke.JOIN_ROUND,
                0, new float[]{10, 10}, 0));
        g.drawLine(getMaxWindowX() / 2, 0, getMaxWindowX() / 2, getMaxWindowY());

        // Draw paddles
        g.setColor(Color.WHITE);
        g.fillRect(leftPaddle.getX(), leftPaddle.getY(),
                leftPaddle.getWidth(), leftPaddle.getHeight());
        g.fillRect(rightPaddle.getX(), rightPaddle.getY(),
                rightPaddle.getWidth(), rightPaddle.getHeight());

        // Draw ball
        g.fillOval(ball.getX(), ball.getY(), ball.getSize(), ball.getSize());

        // Draw scores for both players
        g.setFont(new Font("Arial", Font.BOLD, 48));
        FontMetrics fm = g.getFontMetrics();
        String leftScoreStr = String.valueOf(leftScore);
        String rightScoreStr = String.valueOf(rightScore);
        g.drawString(leftScoreStr,
                getMaxWindowX() / 4 - fm.stringWidth(leftScoreStr) / 2, 80);
        g.drawString(rightScoreStr,
                3 * getMaxWindowX() / 4 - fm.stringWidth(rightScoreStr) / 2, 80);

        // Draw control instructions
        g.setFont(new Font("Arial", Font.PLAIN, 16));
        g.drawString("Player 1: W/S", 20, getMaxWindowY() - 60);
        g.drawString("Player 2: ↑/↓", getMaxWindowX() - 120, getMaxWindowY() - 60);

        // Draw game over overlay and winner message
        if (gameOver) {
            g.setColor(new Color(0, 0, 0, 180)); // Semi-transparent overlay
            g.fillRect(0, 0, getMaxWindowX(), getMaxWindowY());

            g.setColor(Color.WHITE);
            g.setFont(new Font("Arial", Font.BOLD, 72));
            FontMetrics winFm = g.getFontMetrics();
            g.drawString(winner,
                    getMaxWindowX() / 2 - winFm.stringWidth(winner) / 2,
                    getMaxWindowY() / 2 - 50);

            g.setFont(new Font("Arial", Font.PLAIN, 24));
            String restartMsg = "Press R to restart";
            FontMetrics restartFm = g.getFontMetrics();
            g.drawString(restartMsg,
                    getMaxWindowX() / 2 - restartFm.stringWidth(restartMsg) / 2,
                    getMaxWindowY() / 2 + 30);
        }
    }
}