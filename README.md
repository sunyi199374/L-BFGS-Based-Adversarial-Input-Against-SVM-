# L-BFGS-Based-Adversarial-Input-Against-SVM-

Data Source

Method
\section{L-BFGS Based Adversarial Attack}

L-BFGS method is developed from Newton's method. Compared with Newton's method, L-BFGS optimization achieves resource-efficient in both computation and storage.

\subsection{Newton's Method}
To minimize the target function $f(x)$, the second order in Taylor formula of $f(x)$ is computed,
\begin{equation}
       \phi(x)=f(x_k)+\nabla f(x_k)(x-x_k)+\frac{1}{2}(x-x_k)^T\nabla^2  f(x_k)(x-x_k)
\end{equation}
When $f(x)$ reaches its optimum, the Jacobian matrix should be zero,
\begin{equation}
       \nabla \phi(x)=0
\end{equation}
which means
\begin{equation}
       g_k+H_k(x-x_k)=0
\end{equation}
With an initial search point, it is always possible to find a proper search direction and reach the optimum through an iteration process.
\begin{equation}
       x_{k+1}=x_k-a*H_k^{-1}g_k
\end{equation}
Although Newton's method is simple in theory, it is difficult to apply in practice when the order of $x$ is very high. Because,
\begin{itemize}
      \item calculation is too complicated for high-dimensional $H^{-1}$.
      \item the storage of $H^{-1}$ is too large.
  \end{itemize}
  
\subsection{BFGS Method}
In order to reduce calculation, BFGS method approximate $H^{-1}$ using vectors $s_k$ and $y_k$, where
\begin{equation}
s_k=x_{k+1}-x_{k},\ y_k=g_{k+1}-g_k
\end{equation}
It is easy to prove that
\begin{equation}
y_k\approx D_{k+1}s_k
\end{equation}
where $D$ is an approximation of $H^{-1}$.
To calculate $D$, further derive Jacobian matrix $\Delta D_k$
\begin{equation}
\Delta D_k=\frac{s_ks_k^T}{s_k^Ty_k}-\frac{D_ky_ky_k^TD_k}{y_k^TD_ky_k}
\end{equation}
\begin{equation}
D_{k+1}=D_k+\Delta D_k
\end{equation}
Follow the steps above, it's efficient to calculate $D$ through an iterative process with low cost of calculation complexity. 

However, the storage required by BFGS is still large. To store all the parameters, the storage scale is $O(N^2)$.

\subsection{L-BFGS Method}
To further reduce the storage amount, L-BFGS (limit-memory BFGS) method stores only $s_k$ vectors and $y_k$ vectors instead of $D$ matrix. Another difference is that only the latest $m$ pairs of $[s_k, y_k]$ are stored, $m\ll N$ . In this way the storage scale can be reduced to $O(mN)$.


Parameters

Result 

Conclusion
