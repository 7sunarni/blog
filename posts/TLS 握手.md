# tls 证书文件

[https://crypto.stackexchange.com/questions/43697/what-are-the-differences-between-pem-csr-key-crt-and-other-such-file-exte](https://crypto.stackexchange.com/questions/43697/what-are-the-differences-between-pem-csr-key-crt-and-other-such-file-exte)
## pem file
.pem stands for PEM, Privacy Enhanced Mail; it simply indicates a base64 encoding with header and footer lines. Mail traditionally only handles text, not binary which most cryptographic data is, so some kind of encoding is required to make the contents part of a mail message itself (rather than an encoded attachment). The contents of the PEM are detailed in the header and footer line - .pem itself doesn't specify a data type - just like .xml and .html do not specify the contents of a file, they just specify a specific encoding;
## key file
.key can be any kind of key, but usually it is the private key - OpenSSL can wrap private keys for all algorithms (RSA, DSA, EC) in a generic and standard PKCS#8 structure, but it also supports a separate 'legacy' structure for each algorithm, and both are still widely used even though the documentation has marked PKCS#8 as superior for almost 20 years; both can be stored as DER (binary) or PEM encoded, and both PEM and PKCS#8 DER can protect the key with password-based encryption or be left unencrypted;
## csr file 
.csr or .req or sometimes .p10 stands for Certificate Signing Request as defined in PKCS#10; it contains information such as the public key and common name required by a Certificate Authority to create and sign a certificate for the requester, the encoding could be PEM or DER (which is a binary encoding of an ASN.1 specified structure);
## crt file
.crt or .cer stands simply for certificate, usually an X509v3 certificate, again the encoding could be PEM or DER; a certificate contains the public key, but it contains much more information (most importantly the signature by the Certificate Authority over the data and public key, of course).


## TLS 握手
翻译 https://cloudflare.com/learning/ssl/what-happens-in-a-tls-handshake

*在进行 TLS 握手的时候，已经建立好了 TCP 连接，因此在此之前，已经进行了 TCP 的三次握手*

1. Client Hello: 客户端通过发送一条 "hello" 消息给服务端来初始化握手。这条消息中包含了客户端支持的 TLS 版本，加密组（cipher suites）? 支持和一个由客户端随机生成的字符串。

2. Server Hello: 在服务端回复客户端的消息中，包含了服务端的 SSL 证书，服务端选择的密码组和服务端生成的随机字符串。

3. 客户端认证: 客户端通过服务端携带的证书信息验证服务端。这样确认了服务端声称的他是谁，客户端之后会和实际的域名交互。

4. 预主秘钥(premaster secret): 客户端发送一条或者多条随机加密字符串（预主秘钥），预主秘钥是由服务端的 SSL 公钥加密的，只能被存在于服务端的私钥来解密。（客户端从服务端的 SSL 证书中获得的公钥）。

5. 私钥使用: 服务端使用私钥来解密预主秘钥。

6. 创建会话秘钥(Session Key): 客户端和服务端两端都通过客户端随机字符，服务端随机字符，预主秘钥创建会话秘钥。两者创建出来的应该相同。

7. 客户端就绪: 客户端发送用会话秘钥加密的 "finish"。

8. 服务端就绪: 服务端发送用会话秘钥加密的 "finish"。

9. 安全对称加密: TLS 握手结束，服务端客户端继续用会话秘钥沟通。

## cipher suite

cipher suite 是一系列用于建立安全连接的加密算法集合。（一种加密算法是一系列在数据上的数学操作，用于将数据表现得随机化）。现在有一大堆 cipher suite 被广泛使用，成为TLS 握手必要的一部分。
