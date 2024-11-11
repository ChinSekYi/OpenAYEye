import { Box, Typography, useTheme } from "@mui/material";
import { Header } from "../../components";
import { DataGrid } from "@mui/x-data-grid";
import { mockDataTeam } from "../../data/mockData";
import { tokens } from "../../theme";
import {
  AdminPanelSettingsOutlined,
  LockOpenOutlined,
  SecurityOutlined,
} from "@mui/icons-material";
import React, {useState, useEffect, useContext} from "react";
  

// export const mockDataTeam = [
//   {
//     id: 1,
//     surname: "Jon Snow",
//     creditscore: "jonsnow@gmail.com",
//     age: 35,
//     phone: "(665)121-5454",
//     access: "admin",
//   },
// ]
const Team = () => {
  
  // Fetch team data from API
  const [entries, setEntries] = useState([]);
  setEntries(mockDataTeam);
  // useEffect(() => {
  //   // fetch("/getTeams")
  //   fetch("http://localhost:8000/getChurn")
  //   .then((res) => res.json())
  //   .then((data) => {
  //     // console.log(data);
  //     setEntries(data);
  //   });
  // }, []);

  // console.log(entries)

  const theme = useTheme();
  const colors = tokens(theme.palette.mode);

  const columns = [
    { field: "id", headerName: "ID" },
    {
      field: "surname",
      headerName: "Surname",
      flex: 1,
      cellClassName: "name-column--cell",
    },
    { field: "creditscore", headerName: "Credit Score", flex: 1 },
    {
      field: "age",
      headerName: "Age",
      type: "number",
      headerAlign: "left",
      align: "left",
    },
    { field: "gender", headerName: "Gender", flex: 1},
    { field: "gender", headerName: "Gender", flex: 1},
    { field: "exited", headerName: "Exited", flex: 1 },
    // { field: "geography", headerName: "Email", flex: 1 },
    // {
    //   field: "access",
    //   headerName: "Access Level",
    //   flex: 1,
    //   renderCell: ({ row: { access } }) => {
    //     return (
    //       <Box
    //         width="120px"
    //         p={1}
    //         display="flex"
    //         alignItems="center"
    //         justifyContent="center"
    //         gap={1}
    //         bgcolor={
    //           access === "admin"
    //             ? colors.greenAccent[600]
    //             : colors.greenAccent[700]
    //         }
    //         borderRadius={1}
    //       >
    //         {access === "admin" && <AdminPanelSettingsOutlined />}
    //         {access === "manager" && <SecurityOutlined />}
    //         {access === "user" && <LockOpenOutlined />}
    //         <Typography textTransform="capitalize">{access}</Typography>
    //       </Box>
    //     );
    //   },
    // },
  ];


  return (
    <Box m="20px">
      <Header title="Chrun Dataset" subtitle="Dataset Regarding Users Exiting the Bank" />
      <Box
        mt="40px"
        height="75vh"
        flex={1}
        sx={{
          "& .MuiDataGrid-root": {
            border: "none",
          },
          "& .MuiDataGrid-cell": {
            border: "none",
          },
          "& .name-column--cell": {
            color: colors.greenAccent[300],
          },
          "& .MuiDataGrid-columnHeaders": {
            backgroundColor: colors.blueAccent[700],
            borderBottom: "none",
          },
          "& .MuiDataGrid-virtualScroller": {
            backgroundColor: colors.primary[400],
          },
          "& .MuiDataGrid-footerContainer": {
            borderTop: "none",
            backgroundColor: colors.blueAccent[700],
          },
          "& .MuiCheckbox-root": {
            color: `${colors.greenAccent[200]} !important`,
          },
          "& .MuiDataGrid-iconSeparator": {
            color: colors.primary[100],
          },
        }}
      >
        <DataGrid
          rows={entries}
          columns={columns}
          initialState={{
            pagination: {
              paginationModel: {
                pageSize: 10,
              },
            },
          }}
          checkboxSelection
        />
      </Box>
    </Box>
  );
};

export default Team;
