# 使用Node.js 14作为基础镜像
FROM node:14

# 设置工作目录
WORKDIR /app

# 复制package.json和package-lock.json文件
COPY package*.json ./

# 安装依赖
RUN npm install

# 复制所有文件到容器中
COPY . .

# 暴露容器的8080端口
EXPOSE 8080

# 启动Vue服务
CMD ["npm", "run", "serve"]

