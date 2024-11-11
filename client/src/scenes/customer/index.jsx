import {
  Box,
  Button,
  IconButton,
  Typography,
  useMediaQuery,
  useTheme,
  List, 
  ListItem,
  ListItemButton,
  ListItemText, 
} from "@mui/material";
import {
  Header,
  StatBox,
  LineChart,
  ProgressCircle,
  BarChart,
  GeographyChart,
} from "../../components";
import {
  DownloadOutlined,
  Email,
  PersonAdd,
  PointOfSale,
  NotInterested,
  AddReaction, 
  Traffic,
} from "@mui/icons-material";
import { tokens } from "../../theme";
import { mockTransactions } from "../../data/mockData";
import React, {useState, useEffect, useContext} from "react";
import { api } from "../API/backend";

function Customer() {
  const theme = useTheme();
  const colors = tokens(theme.palette.mode);
  const isXlDevices = useMediaQuery("(min-width: 1260px)");
  const isMdDevices = useMediaQuery("(min-width: 724px)");
  const isXsDevices = useMediaQuery("(max-width: 436px)");
  const [totalTraffic, setTraffic] = useState([]);
  const [convertedClients, setConverted] = useState([]);
  const [potentialCustomers, setPotential] = useState([]);
  const [conversionRate, setRate] = useState([]);
  const [segByIncome, setSegIncome] = useState([]);
  const [segByAge, setSegAge] = useState([]);
  const [churnBySeg, setChurnSeg] = useState([]);


  useEffect(() => {
    api.get('totalTraffic')
      .then((res) => res.data.data)
      .then((data) => {
        setTraffic(data);
    });
  }, []);
  // console.log(totalTraffic)

  useEffect(() => {
    api.get('convertedClients')
      .then((res) => res.data.data)
      .then((data) => {
        setConverted(data);
    });
  }, []);

  useEffect(() => {
    api.get('potentialCustomers')
      .then((res) => res.data.data)
      .then((data) => {
        setPotential(data);
    });
  }, []);

  useEffect(() => {
    api.get('conversionRate')
      .then((res) => res.data.data)
      .then((data) => {
        setRate(data);
    });
  }, []);

  useEffect(() => {
    api.get('segByIncome')
      .then((res) => res.data.data)
      .then((data) => {
        setSegIncome(data);
    });
  }, []);

  useEffect(() => {
    api.get('segByAge')
      .then((res) => res.data.data)
      .then((data) => {
        setSegAge(data);
    });
  }, []);

  useEffect(() => {
    api.get('churnBySeg')
      .then((res) => res.data.data)
      .then((data) => {
        setChurnSeg(data);
    });
  }, []);

  // console.log(segByIncome)
    
  return (
    <Box m="20px">
      <Box display="flex" justifyContent="space-between">
        <Header title="Customer Segmentation" subtitle="Welcome to your dashboard" />
        {!isXsDevices && (
          <Box>
            <Button
              variant="contained"
              sx={{
                bgcolor: colors.blueAccent[700],
                color: "#fcfcfc",
                fontSize: isMdDevices ? "14px" : "10px",
                fontWeight: "bold",
                p: "10px 20px",
                mt: "18px",
                transition: ".3s ease",
                ":hover": {
                  bgcolor: colors.blueAccent[800],
                },
              }}
              startIcon={<DownloadOutlined />}
            >
              DOWNLOAD REPORTS
            </Button>
          </Box>
        )}
      </Box>

        {/* GRID & CHARTS */}
        <Box
          display="grid"
          gridTemplateColumns={
            isXlDevices
              ? "repeat(12, 1fr)"
              : isMdDevices
              ? "repeat(6, 1fr)"
              : "repeat(3, 1fr)"
          }
          gridAutoRows="140px"
          gap="20px"
        >
          {/* Statistic Items */}
          <Box
            gridColumn="span 3"
            bgcolor={colors.primary[400]}
            display="flex"
            alignItems="center"
            justifyContent="center"
          >
            <StatBox
              title={ totalTraffic }
              subtitle="Traffic Received"
              // progress="0.19036168720569081"
              // increase="+14%"
              icon={
                <Traffic
                  sx={{ color: colors.greenAccent[600], fontSize: "26px" }}
                />
              }
            />
          </Box>
          <Box
            gridColumn="span 3"
            backgroundColor={colors.primary[400]}
            display="flex"
            alignItems="center"
            justifyContent="center"
          >
            <StatBox
              title={ convertedClients }
              subtitle="Converted Clients"
              // progress="0.50"
              // increase="+21%"
              icon={
                <AddReaction
                  sx={{ color: colors.greenAccent[600], fontSize: "26px" }}
                />
              }
            />
          </Box>
          <Box
            gridColumn="span 3"
            backgroundColor={colors.primary[400]}
            display="flex"
            alignItems="center"
            justifyContent="center"
          >
            <StatBox
              title={ potentialCustomers }
              subtitle="Potential Clients"
              // progress="0.30"
              // increase="+5%"
              icon={
                <PersonAdd
                  sx={{ color: colors.greenAccent[600], fontSize: "26px" }}
                />
              }
            />
          </Box>
          <Box
            gridColumn="span 3"
            backgroundColor={colors.primary[400]}
            display="flex"
            alignItems="center"
            justifyContent="center"
          >
            <StatBox
              title={ conversionRate }
              subtitle="Click Through Rate"
              // progress="0.80"
              // increase="+43%"
              icon={
                <Traffic
                  sx={{ color: colors.greenAccent[600], fontSize: "26px" }}
                />
              }
            />
          </Box>
          <Box
            gridColumn={
              isXlDevices ? "span 12" : isMdDevices ? "span 6" : "span 4"
            }
            gridRow="span 2"
            bgcolor={colors.primary[400]}
          >
            <Box
              mt="25px"
              px="30px"
              display="flex"
              justifyContent="space-between"
            >
              <Box>
                <Typography
                  variant="h5"
                  fontWeight="600"
                  color={colors.gray[100]}
                >
                  Transactions By Income 
                </Typography>
                {/* <Typography
                  variant="h5"
                  fontWeight="bold"
                  color={colors.greenAccent[500]}
                >
                  $59,342.32
                </Typography> */}
              </Box>
              <IconButton>
                <DownloadOutlined
                  sx={{ fontSize: "26px", color: colors.greenAccent[500] }}
                />
              </IconButton>
            </Box>
            <Box height="250px" mt="-20px">
              <BarChart 
              data = { segByIncome }
              index = {"income_cat"}
              keys =  {['Hibernating', 'At Risk', 'Loyal Customers', 'New Customers',]}
              isDashboard={false} />
            </Box>
            
          </Box>
          <Box
            gridColumn={
              isXlDevices ? "span 12" : isMdDevices ? "span 6" : "span 4"
            }
            gridRow="span 2"
            bgcolor={colors.primary[400]}
          >
            <Box
              mt="25px"
              px="30px"
              display="flex"
              justifyContent="space-between"
            >
              <Box>
                <Typography
                  variant="h5"
                  fontWeight="600"
                  color={colors.gray[100]}
                >
                  Transactions Per Age
                </Typography>
                {/* <Typography
                  variant="h5"
                  fontWeight="bold"
                  color={colors.greenAccent[500]}
                >
                  $59,342.32
                </Typography> */}
              </Box>
              <IconButton>
                <DownloadOutlined
                  sx={{ fontSize: "26px", color: colors.greenAccent[500] }}
                />
              </IconButton>
            </Box>
            <Box height="250px" mt="-20px">
              <BarChart 
              data = { segByAge }
              index = {"age_cat"}
              keys =  {['Hibernating', 'At Risk', 'Loyal Customers', 'New Customers',]}
              isDashboard={false} />
            </Box>
            
          </Box>
          <Box
            gridColumn={
              isXlDevices ? "span 12" : isMdDevices ? "span 6" : "span 4"
            }
            gridRow="span 2"
            bgcolor={colors.primary[400]}
          >
            <Box
              mt="25px"
              px="30px"
              display="flex"
              justifyContent="space-between"
            >
              <Box>
                <Typography
                  variant="h5"
                  fontWeight="600"
                  color={colors.gray[100]}
                >
                  Transactions Per Age
                </Typography>
                {/* <Typography
                  variant="h5"
                  fontWeight="bold"
                  color={colors.greenAccent[500]}
                >
                  $59,342.32
                </Typography> */}
              </Box>
              <IconButton>
                <DownloadOutlined
                  sx={{ fontSize: "26px", color: colors.greenAccent[500] }}
                />
              </IconButton>
            </Box>
            <Box height="250px" mt="-20px">
              <BarChart 
              data = { churnBySeg }
              index = {"Segment"}
              keys =  {['Remain', 'Exited']}
              isDashboard={false} />
            </Box>
            
          </Box>
           
        </Box>
    </Box>
  );
}

export default Customer;