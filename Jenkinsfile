pipeline {

  agent any

  stages {

    stage('Checkout Source') {
      steps {
        git url:'https://github.com/sachinj77/myapp2.git', branch:'master'
      }
    }

    stage('Deploy App') {
      steps {
        script {
          kubernetesDeploy(configs: "report-str.yml", kubeconfigId: "kubcfg")
        }
      }
    }

  }

}
