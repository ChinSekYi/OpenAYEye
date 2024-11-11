import apisauce from 'apisauce'

// ? REST API functions to communicate with your database backend
// ? Machine IP - replace with your server's IP address; run `ifconfig` and take the first inet IP address (should be below ens32)
const machineIP = "127.0.0.1"
// const machineIP = "172.25.77.198"
const machinePort = "8000"
export const api = apisauce.create({
    baseURL: `http://${machineIP}:${machinePort}`,
})

